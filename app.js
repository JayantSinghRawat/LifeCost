// app.js
const express = require('express');
const path = require('path');
const engine = require('ejs-mate');
const Joi = require('joi');
const db = require('./config/db');
const errorHandler = require('./middleware/errorHandler');

const app = express();
const PORT = process.env.PORT || 3000;

// Set up ejs-mate as EJS template engine
app.engine('ejs', engine);
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Static files and middleware
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Log application state
console.log('App starting. Database mode:', db.isMock() ? 'Local JSON Fallback' : 'Active MySQL Pool');

// ── 1. DASHBOARD / HOME ROUTE ───────────────────────────────────────────────
app.get('/', async (req, res, next) => {
  try {
    const [groceries] = await db.query('SELECT category, name, quantity_raw, price FROM groceries LIMIT 20');
    res.render('index', { 
      title: 'Life Cost — Dashboard',
      groceries
    });
  } catch (err) {
    next(err);
  }
});

// ── 2. RENT ROUTE ─────────────────────────────────────────────────────────────
app.get('/rent', async (req, res, next) => {
  try {
    // If search inputs are provided, validate and query
    if (req.query.locality) {
      // Joi Validation schema for rent query parameters
      const schema = Joi.object({
        locality: Joi.string().min(2).max(100).required().label('Locality Name'),
        bhk: Joi.number().integer().min(1).max(6).optional().allow('').label('BHK Preference')
      });

      const { error, value } = schema.validate(req.query);
      if (error) {
        return next(error);
      }

      let sql = 'SELECT locality, bhk, bathrooms, sqft, city, rent FROM rent_listings WHERE locality LIKE ?';
      const params = [`%${value.locality}%`];

      if (value.bhk) {
        sql += ' AND bhk = ?';
        params.push(value.bhk);
      }

      sql += ' ORDER BY rent ASC LIMIT 50';

      const [listings] = await db.query(sql, params);
      
      return res.render('rent', {
        title: `Rent in ${value.locality}`,
        listings,
        query: value
      });
    }

    // Default empty search page
    res.render('rent', { title: 'Rent Lookup' });
  } catch (err) {
    next(err);
  }
});

// ── 3. METRO ROUTE ────────────────────────────────────────────────────────────
app.get('/metro', async (req, res, next) => {
  try {
    if (req.query.station) {
      // Joi validation for station query
      const schema = Joi.object({
        station: Joi.string().min(2).max(100).required().label('Station Name')
      });

      const { error, value } = schema.validate(req.query);
      if (error) {
        return next(error);
      }

      const sql = 'SELECT route_name, distance_km, num_stops, travel_min, fare_inr FROM metro_routes WHERE route_name LIKE ? LIMIT 50';
      const [routes] = await db.query(sql, [`%${value.station}%`]);

      return res.render('metro', {
        title: `Metro Routes for ${value.station}`,
        routes,
        query: value
      });
    }

    res.render('metro', { title: 'Metro Fares' });
  } catch (err) {
    next(err);
  }
});

// ── 4. LOCALITY MATCHING ROUTE ────────────────────────────────────────────────
app.get('/locality', async (req, res, next) => {
  try {
    if (req.query.salary || req.query.workspace) {
      // Joi validation for salary and workspace
      const schema = Joi.object({
        salary: Joi.number().positive().min(1000).required().label('Monthly Net Salary'),
        workspace: Joi.string().min(2).max(100).required().label('Workplace Area')
      });

      const { error, value } = schema.validate(req.query);
      if (error) {
        return next(error);
      }

      // Query localities and their average rents
      const [localities] = await db.query(
        'SELECT locality, AVG(rent) as avgRent FROM rent_listings GROUP BY locality ORDER BY avgRent ASC'
      );

      // Score and rank matches based on salary budget cap (25% rule) and distance proximity proxy
      const matches = localities.map(loc => {
        const avgRentVal = Math.round(Number(loc.avgRent || 0));
        const isNearby = loc.locality.toUpperCase().includes(value.workspace.toUpperCase()) || 
                         value.workspace.toUpperCase().includes(loc.locality.toUpperCase());
        return {
          locality: loc.locality,
          avgRent: avgRentVal,
          isNearby
        };
      });

      return res.render('locality', {
        title: 'Locality Matcher Results',
        matches,
        query: value
      });
    }

    res.render('locality', { title: 'Locality Finder' });
  } catch (err) {
    next(err);
  }
});

// ── ERROR HANDLING MIDDLEWARE ────────────────────────────────────────────────
app.use(errorHandler);

// Start Server
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
