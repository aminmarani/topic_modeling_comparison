var express = require("express");
var html = require("html");
var app = express(); //start express
var path = require("path");
var fs = require("fs");
// var csvToJson = require('convert-csv-to-json');

//load index.html
var publicPath = path.resolve(__dirname, "./");
app.use(express.static(publicPath));

var document_text = {};


//convert the csv file to json and put in the data
// function read_in_document_text()
// {
//     var outputFile = 'data/EdEntries.js';
//     csvToJson.fieldDelimiter(',').generateJsonFileFromCsv('data/Processed_ed.csv', outputFile);
//     var content = csvToJson.fieldDelimiter(',').getJsonFromCsv("data/Processed_ed.csv");
//     var count = 0;
    
//     for(var i = 0; i < content.length; i++)
//     {
//         var this_doc = content[i];
//             data = {};
//             data.date = this_doc["date"];
//             data.text = this_doc["entry"];
//             data.url = this_doc["Link"];
//             data.title = this_doc["date"];
//             document_text[count] = data;
//             count++;
//     }
// }

// //ajax endpoint to get the document text we parsed
// app.get("/document_text", function(req, res) {
//     var doc = req.query.doc;
//     // console.log("Serving document " + doc);
//     if(document_text[doc])
//     {
//         res.send(document_text[doc]);
//     }
//     else
//     {
//         res.send("{\"error\":\"error\"}");
//     }
// });

// app.get('/', function(req, res){
//     res.render('index.html',{email:"amh418",password:"000000"});
// });

app.listen(9000, function() {
    // read_in_document_text();
    console.log("App started on http://localhost:9000/");
});