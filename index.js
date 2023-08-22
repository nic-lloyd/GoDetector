var Game = require("tenuki").Game;
const fs = require("fs");

function read_file_into_string(file_path) {
  try {
    const content = fs.readFileSync(file_path, "utf8");
    return content;
  } catch (error) {
    console.log("An error occurred while reading the file:", error);
    return null;
  }
}

var data = JSON.parse(read_file_into_string("data.txt").replace(/'/g, '"'));
var game = new Game({ scoring: "area", boardSize: 13 });

for (d of data) {
  if (game.currentPlayer() !== d[2]) {
    game.pass();
  }
  game.playAt(d[1], d[0]);
}
game.pass();
game.pass();
console.log(game.score());
