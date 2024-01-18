const express = require("express");
const app = express();
const port = 3000;

// Mock database
const database = {
  users: [
    { id: 1, name: "Alice", email: "alice@example.com" },
    { id: 2, name: "Bob", email: "bob@example.com" },
  ],
};

// Async function to simulate database fetching
const fetchUserFromDatabase = async (id) => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const user = database.users.find((user) => user.id === id);
      if (user) {
        resolve(user);
      } else {
        reject(new Error("User not found"));
      }
    }, 1000);
  });
};

// Middleware to parse JSON requests
app.use(express.json());

// GET endpoint to fetch a user
app.get("/user/:id", async (req, res) => {
  try {
    const user = await fetchUserFromDatabase(parseInt(req.params.id));
    res.json(user);
  } catch (error) {
    res.status(404).send(error.message);
  }
});

// POST endpoint to create a new user
app.post("/user", (req, res) => {
  const newUser = {
    id: database.users.length + 1,
    name: req.body.name,
    email: req.body.email,
  };
  database.users.push(newUser);
  res.status(201).json(newUser);
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
