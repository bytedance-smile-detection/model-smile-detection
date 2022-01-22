const express = require("express");
const path = require("path");
const cors = require("cors");

const app = express();

app.use(cors());
folderPath = path.resolve(__dirname, "..");
app.use(express.static(folderPath + "/models"));

app.listen(8000, () => {
  console.log("服务已启动，正在监听8000端口 ...");
});
