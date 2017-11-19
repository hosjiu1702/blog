var mosca = require('mosca')
var http = require('http')
var fs = require('fs')
var url = require('url')
//var connect = require('connect')
//var serveStatic = require('serve-static')

//var app = connect()

//Use middleware
//app.use(serveStatic(__dirname))

var http_port = 3000

var settings = {
  port: 1884,
  backend: {
    type: 'mongo',
    url: 'mongodb://localhost:27017/mqtt_hosjiu'
  },
  persistence: {
    factory: mosca.persistence.Mongo,
    url: 'mongodb://localhost:27017/mqtt_hosjiu'
  }
}
  
/*Create mqtt server*/
var broker_server = new mosca.Server(settings)
/*Create http server*/
var http_server = http.createServer(function(req, res){
  //Parse url using 'Url' built-in module
  var _url = url.parse(req.url, true)

  console.log('createServer() is invoked')
  // -- DASHBOARD --
  if(_url.pathname === '/dashboard'){
    //Return a dashboard html page for browser
    fs.readFile( __dirname + '/dashboard.html', 'utf8', function(err, data){
      if(err){
        res.writeHead(404)
        res.end()
        throw err
      }
      res.writeHead(200, {'Content-Type': 'text-html'})
      res.write(data)
      res.end()

      console.log('dashboard was rendered !? -- DEBUG')
    })
  }
  else{
    res.writeHead(404)
    res.end('Not Found')
  }
})

/*Let http server listens at port 3000*/
http_server.listen(http_port)

/*Attach a exist http server to mqtt server*/
broker_server.attachHttpServer(http_server)

http_server.on('listening', function(){
  console.log('http server is listening at port ' + http_port)
})

broker_server.on('error', function(err){
  console.log(err.message)
  return -1
})

broker_server.on('ready', function(){
  console.log('mqtt server is running at port ' + settings.port)
})

broker_server.on('subscribed', function(topic, client){
  console.log('*--subscribed--* :')
  console.log('topic: ' + topic)
  console.log('id: ' + client.id)
  console.log('')
})

broker_server.on('published', function(packet, client){
  if(client !== undefined){
    console.log('*--published--* :')
    console.log('payload: ' + packet.payload)
    console.log('id: ' + client.id)
    console.log('')

  }
})

broker_server.on('clientDisconnected', function(client){
  console.log('*--clientDisconnected--* : ')
  console.log('id: ' + client.id)
  console.log('')
})

