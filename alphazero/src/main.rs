mod game;
use crate::game::Game;
use crate::game::Connect4Game;

fn main() {

    let mut game = Connect4Game::new();
    assert_eq!(game.get_result(0.0, 0.0), None);
    game.make_move(0.0);
    game.make_move(1.0);
    game.make_move(0.0);
    game.make_move(1.0);
    game.make_move(0.0);
    game.make_move(1.0);
    println!("{:?}", game.make_move(0.0));
    game.make_move(1.0);
}
