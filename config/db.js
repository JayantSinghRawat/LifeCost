// config/db.js
const mysql = require('mysql2/promise');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

let pool = null;
let useMock = false;
let mockData = {
  rent: [],
  metro: [],
  grocery: []
};

// Lazy load mock data from JSON files if MySQL fails
function loadMockData() {
  try {
    const rentPath = path.join(__dirname, '..', 'Scrapping_Manual', 'renting', 'listings.json');
    const metroPath = path.join(__dirname, '..', 'Scrapping_Manual', 'Metro', 'bhopal_metro_fares.json');
    const groceryPath = path.join(__dirname, '..', 'Scrapping_Manual', 'grocery', 'Bhopal_blinkit_results_462010.json');

    if (fs.existsSync(rentPath)) {
      const raw = JSON.parse(fs.readFileSync(rentPath, 'utf8'));
      mockData.rent = raw.map((item, idx) => {
        const details = String(item.details || '');
        const priceStr = String(item.price || '');
        const locationStr = String(item.location || '');
        
        const bhkMatch = details.match(/(\d+)\s*(BHK|RK|BRK)/i);
        const bhk = bhkMatch ? parseInt(bhkMatch[1], 10) : 2;

        const bathMatch = details.match(/(\d+)\s*Bathroom/i);
        const bathrooms = bathMatch ? parseInt(bathMatch[1], 10) : bhk;

        const sqftMatch = details.match(/(\d[\d,]*)\s*sqft/i);
        const sqft = sqftMatch ? parseFloat(sqftMatch[1].replace(/,/g, '')) : 1000;

        const priceMatch = priceStr.replace(/\s/g, '').match(/\d[\d,]*/);
        const rent = priceMatch ? parseFloat(priceMatch[0].replace(/,/g, '')) : 10000;

        const parts = locationStr.split(',').map(p => p.trim().toUpperCase());
        const city = parts[parts.length - 1] || 'BHOPAL';
        const locality = parts[0] || 'OTHER';

        return { id: idx + 1, locality, bhk, bathrooms, sqft, city, rent };
      });
    }

    if (fs.existsSync(metroPath)) {
      const raw = JSON.parse(fs.readFileSync(metroPath, 'utf8'));
      mockData.metro = raw.map((item, idx) => {
        const from = item.From || '';
        const to = item.To || '';
        const route_name = `${from} - ${to}`;
        const distance = parseFloat((item.Distance || '0').replace(/[^\d.]/g, '')) || 0;
        const num_stops = parseInt((item.Stations || '0').replace(/[^\d]/g, ''), 10) || 0;
        const travel_min = parseFloat((item['Est. Time'] || '0').replace(/[^\d.]/g, '')) || 0;
        const fare_inr = parseInt((item['Est. Fare'] || '10').replace(/[^\d]/g, ''), 10) || 10;
        return { id: idx + 1, route_name, distance_km: distance, num_stops, travel_min, fare_inr };
      });
    }

    if (fs.existsSync(groceryPath)) {
      const raw = JSON.parse(fs.readFileSync(groceryPath, 'utf8'));
      mockData.grocery = raw.map((item, idx) => {
        const name = item.name || '';
        const category = item.term || 'Groceries';
        const priceStr = item.price || '';
        const price = parseFloat(priceStr.replace(/[^\d.]/g, '')) || 0;
        return { id: idx + 1, category, name, price };
      });
    }
    
    console.log('Mock database system initialized from local JSON files.');
  } catch (err) {
    console.error('Failed to initialize mock fallback data:', err.message);
  }
}

// Initialise DB pool
try {
  pool = mysql.createPool({
    host: process.env.DB_HOST || '127.0.0.1',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_DATABASE || 'lifecost',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
  });
  
  // Test connection
  pool.getConnection()
    .then(conn => {
      console.log('Successfully connected to MySQL database.');
      conn.release();
    })
    .catch(err => {
      console.warn('MySQL server not reachable. Switching to Local JSON Mock mode.');
      useMock = true;
      loadMockData();
    });
} catch (e) {
  console.warn('Could not initialize MySQL pool. Switching to Local JSON Mock mode.');
  useMock = true;
  loadMockData();
}

/**
 * Custom query wrapper that seamlessly falls back to local data matching
 */
async function query(sql, params = []) {
  if (useMock) {
    const lowerSql = sql.toLowerCase();
    
    // Simulate rent_listings query
    if (lowerSql.includes('rent_listings')) {
      let filtered = [...mockData.rent];
      
      // Basic filtering based on locality or other query params
      if (params.length > 0) {
        const paramVal = params[0];
        if (typeof paramVal === 'string') {
          // locality query
          filtered = filtered.filter(item => item.locality.includes(paramVal.toUpperCase()));
        }
      }
      return [filtered];
    }
    
    // Simulate metro_routes query
    if (lowerSql.includes('metro_routes')) {
      let filtered = [...mockData.metro];
      if (params.length > 0) {
        const paramVal = params[0];
        if (typeof paramVal === 'string') {
          filtered = filtered.filter(item => item.route_name.toLowerCase().includes(paramVal.toLowerCase()));
        }
      }
      return [filtered];
    }

    // Simulate groceries query
    if (lowerSql.includes('groceries')) {
      return [mockData.grocery];
    }

    return [[]];
  }

  // Real MySQL execution
  const [rows] = await pool.execute(sql, params);
  return [rows];
}

module.exports = {
  query,
  pool,
  isMock: () => useMock
};
