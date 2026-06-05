// traffic-trigger: Cloudflare Worker
// Receives GET from cron-job.org, validates secret, dispatches GitHub Actions workflow_dispatch
// Secrets required: CRON_SECRET, GITHUB_TOKEN
//
// Two cron jobs configured at cron-job.org:
// - "TraffiCOracle Dedup" - runs at 3am daily, calls with ?type=dedup
// - "TraffiCOracle Snapshot" - runs twice/hour, calls with ?type=snapshot

const REPO = 'thecont1/traffic-monitor-lizard';
const WORKFLOW = 'traffic_snapshot.yml';
const BRANCH = 'main';

export default {
  async fetch(request, env) {
    // Only allow GET requests
    if (request.method !== 'GET') {
      return new Response('Method Not Allowed', { status: 405 });
    }

    // Validate the shared secret sent by cron-job.org
    const incomingSecret = request.headers.get('X-Cron-Secret');
    if (!incomingSecret || incomingSecret !== env.CRON_SECRET) {
      console.log('Unauthorized request - bad or missing X-Cron-Secret');
      return new Response('Unauthorized', { status: 401 });
    }

    // Determine job type from query parameter
    const url = new URL(request.url);
    const jobType = url.searchParams.get('type') || 'snapshot';
    if (jobType !== 'snapshot' && jobType !== 'dedup') {
      return new Response('Invalid job type', { status: 400 });
    }

    // Dispatch the GitHub Actions workflow_dispatch event
    const apiUrl = `https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`;

    const dispatchBody = JSON.stringify({
      ref: BRANCH,
      inputs: {
        job_type: jobType
      }
    });

    let ghResponse;
    try {
      ghResponse = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${env.GITHUB_TOKEN}`,
          'Accept': 'application/vnd.github+json',
          'Content-Type': 'application/json',
          'User-Agent': 'traffic-trigger/1.0',
        },
        body: dispatchBody,
      });
    } catch (err) {
      console.error('GitHub API fetch error:', err.message);
      return new Response(
        JSON.stringify({ error: 'Failed to reach GitHub API', detail: err.message }),
        { status: 502, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const timestamp = new Date().toISOString();

    if (ghResponse.status === 204) {
      // 204 No Content = GitHub accepted the dispatch
      console.log(`[${timestamp}] workflow_dispatch accepted by GitHub (204), job_type=${jobType}`);
      return new Response(
        JSON.stringify({ status: 'dispatched', job_type: jobType, timestamp }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Anything else is an error from GitHub
    let body = '';
    try { body = await ghResponse.text(); } catch (_) {}
    console.error(`[${timestamp}] GitHub returned ${ghResponse.status}: ${body}`);
    return new Response(
      JSON.stringify({ status: 'github_error', code: ghResponse.status, detail: body, timestamp }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    );
  },
};