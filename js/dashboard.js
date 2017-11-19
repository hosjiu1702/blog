
			function subscribe()
			{
				
				var BrokerMosca = 'ws://127.0.0.1:3000'

				var opts = {
					keepAlive: 60,
					protocolId: 'MQIsdp',
					protocolVersion: 3,
					clientId: 'hosjiu-browser-client',
					clean: true
				}
				
				var topic = document.getElementById("topic-sub").value;
				
				var client = mqtt.connect(BrokerMosca, opts)

				client.on('connect', function(){
					console.log('Connect Events is emitted !')
					client.subscribe(topic, function(err, granted){
						if(err){
							var topic = granted[0]
							var qos = granted[1]
							console.log('subscribe to ' + topic + 'with qos (' + qos + ') ' + 'failed')
							console.log(err)
						}
						console.log('subscribe to ' + topic + 'with qos (' + qos + ') ' + 'successfully')
					});
					client.publish('room/data', 'this is still alright', function(err){
						if(err){
							console.log(err)
						}
						console.log('Published')
					})
				})

				client.on('reconnect', function(){
					console.log('reconnect')
				})

				client.on('error', function(err){
					console.log(err)
				})

				/*
				var msgPub = "I am hosjiu, this is my alias :)";
				client.publish(topic, msgPub);	
				*/

				client.on('message', function(topic, message){
					console.log(message)
					document.getElementById("message").innerHTML = message;
					client.end();
				});
			}

function publish()
{
				var BrokerMosca = 'ws://127.0.0.1:3000'
				var opts = {
					keepAlive: 60,
					protocolId: 'MQIsdp',
					protocolVersion: 3,
					clientId: 'hosjiu-browser-client',
					clean: true
				}
				var topic = document.getElementById("topic-pub").value;
				var client = mqtt.connect(BrokerMosca, opts)

				client.publish(topic, "hello", function(err){
					if(err){
						console.log(err)
						throw err
					}
					console.log('published to ' + topic)
					client.end()
				})
}