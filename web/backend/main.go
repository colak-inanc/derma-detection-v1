package main

import (
	"VeriMadenciligi/database"
	"VeriMadenciligi/routes"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"log"
)

func main() {

	database.Connect()

	app := fiber.New()

	app.Use(cors.New(cors.Config{
		AllowOrigins: "http://localhost:8000",
	}))

	routes.Setup(app)

	log.Fatal(app.Listen(":8000"))
}
