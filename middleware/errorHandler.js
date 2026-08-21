// middleware/errorHandler.js

function errorHandler(err, req, res, next) {
  console.error('Error caught in middleware:', err);

  const status = err.status || err.statusCode || 500;
  const message = err.message || 'Internal Server Error';

  // Handle Joi validation errors specifically
  if (err.isJoi) {
    const details = err.details.map(d => d.message).join(', ');
    res.status(400);
    if (req.xhr || req.headers.accept.indexOf('json') > -1) {
      return res.json({ success: false, error: details });
    }
    return res.render('error', { 
      title: 'Bad Request', 
      statusCode: 400, 
      message: `Validation Error: ${details}` 
    });
  }

  // Handle standard errors
  res.status(status);
  
  if (req.xhr || (req.headers.accept && req.headers.accept.indexOf('json') > -1)) {
    return res.json({ success: false, error: message });
  }

  res.render('error', { 
    title: 'An Error Occurred', 
    statusCode: status, 
    message: message 
  });
}

module.exports = errorHandler;
