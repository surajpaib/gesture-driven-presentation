const ws = WebSocket('ws://localhost:7777/pose');


ws.onopen = function() {
                  
    // Web Socket is connected, send data using send()
    ws.send("Message to send");
    alert("Message is sent...");
 };
  
 ws.onmessage = function (evt) { 
    var received_msg = evt.data;
    alert("Message is received...");
 };
  
 ws.onclose = function() { 
    
    // websocket is closed.
    alert("Connection is closed..."); 
 };


