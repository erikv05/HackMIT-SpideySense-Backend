const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");

const app = express();
app.use(bodyParser.json());
app.use(cors());

app.post("/predict", (req, res) => {
  const data = req.body;

  // Call the Python Flask API
  fetch("http://127.0.0.1:8080/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => response.json())
    .then((result) => {
      const prediction = result.prediction;
      return res.json({ prediction });
    })
    .catch((error) => {
      return res.status(400).json({ error: "Error:", error });
    });
});

const PORT = 8081;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
