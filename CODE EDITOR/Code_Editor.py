<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Editor</title>
    <link rel="stylesheet" href="style.css">
    <script src="https://kit.fontawesome.com/167d741242.js" crossorigin="anonymous"></script>

<style>
*{                                                                                                //CSS
    margin:0;
    padding:0;
font-family: Arial, Helvetica, sans-serif;    box-sizing: border-box;
}
body{
background: linear-gradient(270deg,grey,black);
color : black;
font-size: small;
}
.container{
   width:100%;
   height :100vh;
   padding:20px;
  display: flex;
}
.left, .right{
    flex-basis:50%;
    padding: 10px;;
}
textarea{
   width :100%;
   height:30%;
   background-color: dimgrey;
   color: aliceblue;
   padding: 10px, 20px;
   border:0;
   outline:0;
   font-size: 18px;
}
iframe{
 width :100%;
height:100%;
   background:honeydew;
   border:0;
   outline:0;

}
label i{
   margin-right:10px; 
   margin-left:10px; 
}
label{
    color: white;
    display: flex;
    background-color:black;
    height: 30px;
    align-items: center;
}
</style>
</head>

<body>
   <div class="container">                                                                  //HTML
    <div class="left">
        <label><i class="fa-brands fa-html5"></i> HTML</label>
        <textarea id="htmlc" onkeyup="run()"></textarea>
        <label><i class="fa-brands fa-css3-alt"></i> CSS</label>
        <textarea id="cssc"onkeyup="run()"></textarea>
        <label><i class="fa-brands fa-js"></i> JS</label>
        <textarea id="jsc"onkeyup="run()"></textarea>
    </div>

    <div class="right">
        <label><i class="fa-solid fa-play fa-beat"></i>  OUTPUT</label>
        <iframe id="output"></iframe>
    </div>
   </div>

   <script>                                                                                   //JS
    function run(){
        let htmlcode=document.getElementById("htmlc").value;
        let csscode=document.getElementById("cssc").value;
        let jscode=document.getElementById("jsc").value;
        let output=document.getElementById("output");
        output.contentDocument.body.innerHTML=htmlcode+"<style>"+csscode+"</style>";
        output.contentWindow.eval(jsCode);
    }  
   </script>
</body>
</html>
