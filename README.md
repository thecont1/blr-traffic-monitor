# Bangalore Traffic Monitor 🚗📊

A smart system that automatically monitors and analyzes real-time traffic conditions on key routes throughout Bangalore city. Perfect for commuters, city planners, and anyone curious about traffic patterns in India's Silicon Valley.

## What This Does

Ever wondered which routes in Bangalore have the worst traffic? This system continuously monitors predetermined routes across the city and gives you insights like:

- **Real-time travel times** between popular locations
- **Average speeds** on different routes
- **Traffic patterns** throughout the day
- **Historical data** to spot trends

Think of it as your personal traffic analyst that works 24/7!

## How It Works

The system uses **Google Maps** to check travel times between various locations around Bangalore. It:

1. **Automatically visits** Google Maps for each route
2. **Extracts** current travel time and distance data
3. **Calculates** average speeds and traffic conditions
4. **Stores** everything in CSV files for analysis
5. **Generates** visual reports and insights

All of this happens automatically using web automation - no manual checking required!

## Key Features

### 🗺️ **Smart Location System**
- Uses **Plus Codes** (like postal codes, but more precise) to identify exact locations
- Covers major areas including:
  - Metro stations (MG Road, Majestic, Hebbal)
  - Tech hubs (Embassy TechVillage, Biocon Campus)
  - Airports (Kempegowda International)
  - Shopping areas (Lulu Mall, Nexus Mall)
  - Hospitals, temples, and other landmarks

### 📈 **Data Analysis**
- **Travel duration** tracking (in minutes)
- **Distance measurement** (in kilometers)  
- **Average speed calculation** (km/hour)
- **Time-based analysis** (hourly, daily patterns)
- **Route comparison** (which routes are fastest/slowest)

### 🤖 **Automated Collection**
- Runs completely hands-free using Selenium web automation
- Handles different traffic conditions automatically
- Retry logic for reliable data collection
- Timezone-aware timestamps

### 📊 **Visual Reports**
- Interactive maps showing all monitored routes
- Data visualizations in Jupyter notebooks
- CSV exports for further analysis
- Real-time traffic summaries

## Getting Started

### Prerequisites
- Python 3.13+
- Chrome browser (for web automation)
- Internet connection

### Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/thecont1/blr-traffic-monitor.git
   cd blr-traffic-monitor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if you use UV (modern Python package manager):
   ```bash
   uv sync
   ```

3. **Run the traffic monitor:**
   ```bash
   python traffic_snapshot.py
   ```

### What Happens When You Run It

1. The system opens Chrome in the background (you won't see it)
2. It visits Google Maps for each route in the `csv-routes.csv` file
3. Extracts travel time and distance data
4. Saves results to `csv-bangalore_traffic.csv`
5. Shows you a summary of the worst traffic route found

Example output:
```
21hrs [traffic_snapshot] 51 mins @ 40.35 Km/hr (→ Kempegowda International Airport, Bengaluru)
```

## Understanding the Data

### Route Codes
Routes are identified using Plus Code pairs like `2HM2+P8|XJV5+RG`, which means:
- `2HM2+P8` = Starting location (Jaya Prakash Narayana Park)
- `XJV5+RG` = Ending location (Coles Park, Fraser Town)

### Files Explained

| File | Purpose |
|------|---------|
| `traffic_snapshot.py` | Main script that collects traffic data |
| `csv-routes.csv` | List of routes to monitor |
| `csv-locations_*.csv` | Maps Plus Codes to human-readable location names |
| `csv-bangalore_traffic.csv` | Collected traffic data (created after first run) |
| `traffic_visual.ipynb` | Jupyter notebook for data analysis and visualization |
| `routes.html` & `routes.png` | Visual map of all monitored routes |

### Sample Data Structure
```csv
year,month,date,hour,origin,destination,duration,distance,avg_speed
2025,9,25,21,MG Road Metro Station,Kempegowda International Airport,51,34.3,40.35
```

## Customizing Routes

Want to monitor different routes? Easy!

1. **Add new locations** to `csv-locations_12.9514242_77.6590212.csv`
2. **Add route combinations** to `csv-routes.csv` using Plus Code format
3. **Run the system** - it will automatically monitor your new routes

**Pro tip:** Use [plus.codes](https://plus.codes) to find Plus Codes for any location!

## Analysis & Insights

The included Jupyter notebook (`traffic_visual.ipynb`) helps you:
- Identify consistently slow routes
- Find optimal travel times
- Compare different areas of the city
- Spot traffic patterns and trends

Perfect for:
- **Commuters** planning optimal travel times
- **Delivery services** optimizing routes
- **City planners** understanding traffic flows
- **Researchers** studying urban mobility

## Technical Details

### Libraries Used
- **Selenium**: Web automation for Google Maps
- **Pandas**: Data manipulation and analysis  
- **OpenLocationCode**: Plus Code handling
- **TimezoneFinder**: Accurate timestamp handling

### Architecture
- **Modular design** with separate data collection and analysis
- **Error handling** for reliable data collection
- **Timezone awareness** for accurate timestamps
- **CSV-based storage** for easy data sharing and analysis

## Contributing

Have ideas for improvement? Found a bug? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Uses Google Maps for traffic data
- Built with modern Python tools and best practices
- Inspired by the need to understand Bangalore's complex traffic patterns

---

**Happy traffic monitoring!** 🚦 Now you'll never be surprised by unexpected traffic jams again.