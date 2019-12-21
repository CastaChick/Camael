var express = require('express')
  , http = require('http')
  , app = express()
  ;
app.use(express.static(__dirname + '/docs/build/html'));
var port = process.env.PORT || 5000;
var server = http.createServer(app).listen(port);