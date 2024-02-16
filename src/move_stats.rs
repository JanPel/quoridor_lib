use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::board::{Board, Move, Position, WallDirection};

impl Move {
    fn to_quoridor_strat_notation(&self, board: &Board) -> String {
        let string = match self {
            Move::PawnMove(first_step, second_step) => {
                let current_pawn = board.pawns[board.turn % 2];
                let next_pos = current_pawn.position.add_move(*first_step);
                if let Some(second_step) = second_step {
                    let next_pos = next_pos.add_move(*second_step);
                    format!("{}", next_pos.encode())
                } else {
                    format!("{}", next_pos.encode())
                }
            }
            Move::Wall(dir, pos) => match dir {
                WallDirection::Horizontal => format!("{}h", pos.encode()),
                WallDirection::Vertical => format!("{}v", pos.encode()),
            },
        };
        string.to_lowercase()
    }

    fn from_quoridor_strat_notation(move_str: &str, board: &Board) -> Self {
        let move_str = move_str.to_uppercase();
        if move_str.len() == 3 {
            let pos = Position::decode(&move_str.chars().take(2).collect::<String>()).unwrap();
            let dir = match move_str.chars().nth(2).unwrap() {
                'H' => WallDirection::Horizontal,
                'V' => WallDirection::Vertical,
                _ => panic!("Invalid wall direction"),
            };
            Move::Wall(dir, pos)
        } else {
            let pos = Position::decode(&move_str.chars().take(2).collect::<String>()).unwrap();
            let pawn_move = board
                .is_possible_next_pawn_location(pos.row as usize, pos.col as usize)
                .unwrap();
            Move::PawnMove(pawn_move.0, pawn_move.1)
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MoveStats {
    total_games: u32,
    wins_player_0: u32,
    wins_player_1: u32,
    children: HashMap<String, MoveStats>,
}
const LEFT_WALLS: [char; 4] = ['a', 'b', 'c', 'd'];
const RIGHT_WALLS: [char; 4] = ['h', 'g', 'f', 'e'];

const RIGHT_PAWNS: [char; 4] = ['i', 'h', 'g', 'f'];

fn mirror(move_str: &str, left_moves: &[char; 4], right_moves: &[char; 4]) -> String {
    let mut to_return = move_str.to_string();
    if let Some(index) = right_moves
        .iter()
        .position(|x| *x == move_str.chars().next().unwrap())
    {
        to_return.replace_range(0..1, &left_moves[index].to_string());
        to_return
    } else if let Some(index) = left_moves
        .iter()
        .position(|x| *x == move_str.chars().next().unwrap())
    {
        to_return.replace_range(0..1, &right_moves[index].to_string());
        to_return
    } else {
        to_return
    }
}

fn mirror_wall_move(move_str: &str) -> String {
    mirror(move_str, &LEFT_WALLS, &RIGHT_WALLS)
}

fn mirror_pawn_move(move_str: &str) -> String {
    mirror(move_str, &LEFT_WALLS, &RIGHT_PAWNS)
}

pub fn mirror_move(move_str: &str) -> String {
    if move_str.len() == 3 {
        mirror_wall_move(move_str)
    } else if move_str.len() == 2 {
        mirror_pawn_move(move_str)
    } else {
        panic!("Invalid move string: {}", move_str);
    }
}

pub fn is_mirrored(move_str: &str) -> Option<bool> {
    if move_str.len() == 3 {
        Some(RIGHT_WALLS.contains(&move_str.chars().next().unwrap()))
    } else if move_str.len() == 2 {
        if &move_str.chars().next().unwrap() == &'e' {
            return None;
        }
        Some(RIGHT_PAWNS.contains(&move_str.chars().next().unwrap()))
    } else {
        panic!("Invalid move string: {}", move_str);
    }
}

impl MoveStats {
    pub fn new() -> Self {
        Self {
            total_games: 0,
            wins_player_0: 0,
            wins_player_1: 0,
            children: HashMap::new(),
        }
    }
    fn update(&mut self, moves: &[String], winner_first_player: bool) {
        self.total_games += 1;
        if winner_first_player {
            self.wins_player_0 += 1;
        } else {
            self.wins_player_1 += 1;
        }

        if let Some(next_move) = moves.first() {
            let entry = self
                .children
                .entry(next_move.to_string())
                .or_insert(Self::new());
            entry.update(&moves[1..], winner_first_player);
        }
    }

    fn pretty_print(&self, board: Board) {
        let mut queue = VecDeque::new();
        queue.push_back((self, "".to_string(), 0, board));
        let mut count = 0;
        while let Some((next_to_print, moves_code, depth, mut board)) = queue.pop_back() {
            if next_to_print.total_games <= 20 {
                continue;
            }
            count += 1;
            println!(
                "{} total_games: {}, wins_0: {} wins_1: {}, historic moves: {}",
                board.encode(),
                next_to_print.total_games,
                next_to_print.wins_player_0,
                next_to_print.wins_player_1,
                moves_code
            );
            for (key, value) in &next_to_print.children {
                let game_move = Move::from_quoridor_strat_notation(key, &board);
                let mut board = board.clone();
                board.game_move(game_move);
                queue.push_back((value, format!("{}:{}", moves_code, key,), depth + 1, board));
            }
        }
        println!("{} BOARD WITH MORE THEN 10 visits", count);
    }

    fn best_move_for_player(&self, player: usize) -> Option<(f32, String)> {
        let mut best_move = None;
        let mut best_win_rate = 0.0;
        let mut visits_best = 0;
        for (key, value) in &self.children {
            //if value.total_games < 6 {
            //    continue;
            //}

            let win_rate = if player == 0 {
                value.wins_player_0 as f32 / value.total_games as f32
            } else {
                value.wins_player_1 as f32 / value.total_games as f32
            };
            //if win_rate < 0.45 {
            //    continue;
            //}

            if win_rate >= best_win_rate {
                best_win_rate = win_rate;
                visits_best = value.total_games;
                best_move = Some((win_rate, key.to_string()));
            }
        }
        best_move
    }

    pub fn moves_seen(&self, board: &Board, is_mirrored: bool) -> Vec<(f32, Move)> {
        let mut moves = Vec::new();
        for (key, value) in &self.children {
            let move_str = if is_mirrored {
                mirror_move(key)
            } else {
                key.to_string()
            };
            let move_ = Move::from_quoridor_strat_notation(&move_str, &board);
            let win_rate = if board.turn % 2 == 0 {
                value.wins_player_0 as f32 / value.total_games as f32
            } else {
                value.wins_player_1 as f32 / value.total_games as f32
            };
            moves.push((win_rate, move_));
        }
        moves
    }

    pub fn best_move(&self, board: &Board, is_mirrored: bool) -> Option<(f32, Move)> {
        let (win_rate, best_move) = self.best_move_for_player(board.turn % 2)?;
        let best_move = if is_mirrored {
            Move::from_quoridor_strat_notation(&mirror_move(&best_move), &board)
        } else {
            Move::from_quoridor_strat_notation(&best_move, &board)
        };
        Some((win_rate, best_move))
    }
    fn take_move_int(&mut self, move_str: &str, is_mirrored: bool) {
        let move_str = if is_mirrored {
            mirror_move(move_str)
        } else {
            move_str.to_string()
        };
        if let Some(next_stats) = self.children.get_mut(&move_str) {
            let mut next_stats_swap = MoveStats::new();
            std::mem::swap(next_stats, &mut next_stats_swap);
            *self = next_stats_swap;
        } else {
            *self = Self::new();
        }
    }

    pub fn take_move(&mut self, game_move: Move, is_mirrored: bool, old_board: &Board) {
        let move_str = game_move.to_quoridor_strat_notation(old_board);
        self.take_move_int(&move_str, is_mirrored);
    }
}

// Store the precalcuted scores for player zero in a HashMap;
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreCalc {
    scores_player_zero: HashMap<String, Option<f32>>,
}

impl PreCalc {
    pub fn new() -> Self {
        Self {
            scores_player_zero: HashMap::new(),
        }
    }
    pub fn store(&self, file_name: &str) {
        let json = serde_json::to_string(&self).unwrap();
        std::fs::write(file_name, json).unwrap();
    }
    pub fn load(file_name: &str) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let json = std::fs::read_to_string(file_name)?;
        Ok(serde_json::from_str(&json)?)
    }
    pub fn open(file_name: &str) -> Self {
        if let Ok(precalc) = Self::load(file_name) {
            return precalc;
        } else {
            PreCalc::new()
        }
    }
    fn update_with_move_stats(&mut self, move_stats: &MoveStats, board: Board) {
        let mut queue = VecDeque::new();
        queue.push_back((move_stats, "".to_string(), 0, board));
        while let Some((next_to_print, moves_code, depth, mut board)) = queue.pop_back() {
            if next_to_print.total_games <= 3 || board.turn >= 16 {
                continue;
            }

            // fresh prince eric opening
            //let looking_for = ":e2:e8:e3:e7:e4:e6:d1v";
            // The clasic
            //let looking_for = ":e2:e8:e3:e7:e4:e6:d3h:d7h";
            // The classic classic
            //let looking_for = ":e2:e8:e3:e7:e4:e6:d3h:d7h:e5:f7h:d4h";
            // the godalec variant
            //let looking_for = ":e2:e8:e3:e7:e4:e6:d3h:d7h:e5:f7h";
            let looking_for = ":e2:e8:e3:e7:e4:e6:d3h:d7h";

            //let looking_for = ":e2:e8:e3:e7:e4:e6:d3h:c6h:f3h";
            //if board.encode() == "9;8E4;9E6;D3h;F3h;C6h" {
            //    println!(
            //        "{} total_games: {}, wins_0: {} wins_1: {}, historic moves: {}",
            //        board.encode(),
            //        next_to_print.total_games,
            //        next_to_print.wins_player_0,
            //        next_to_print.wins_player_1,
            //        moves_code
            //    );
            //}
            if moves_code.len() >= looking_for.len()
                && &moves_code[0..looking_for.len()] == looking_for
            {
                println!(
                    "{} total_games: {}, wins_0: {} wins_1: {}, historic moves: {}",
                    board.encode(),
                    next_to_print.total_games,
                    next_to_print.wins_player_0,
                    next_to_print.wins_player_1,
                    moves_code
                );
                self.scores_player_zero.insert(board.encode(), None);
            }
            for (key, value) in &next_to_print.children {
                let game_move = Move::from_quoridor_strat_notation(key, &board);
                let mut board = board.clone();
                board.game_move(game_move);
                queue.push_back((value, format!("{}:{}", moves_code, key,), depth + 1, board));
            }
        }
    }
    pub fn insert_result(&mut self, board: &Board, result: f32) {
        self.scores_player_zero.insert(board.encode(), Some(result));
    }

    pub fn roll_out_score(&self, board: &Board) -> Option<f32> {
        if let Some(score) = self.scores_player_zero.get(&board.encode()) {
            *score
        } else {
            // We want to encode the mirrored board as well
            if let Some(score) = self.scores_player_zero.get(&board.encode_mirror()) {
                *score
            } else {
                None
            }
        }
    }

    // We calculate from boards further progessed into the game back. This way we can reuse the deeper calculations later on. If we hit one of those already calculated board with monte carlo
    pub fn next_to_calc(&self) -> Option<Board> {
        let mut boards = vec![];
        for (key, value) in &self.scores_player_zero {
            if value.is_none() {
                boards.push(Board::decode(key).unwrap());
            }
        }
        boards.sort_by_key(|x| x.turn);
        boards.pop()
    }

    // Here we return all the boards that haven't been precalculated yet at the deepest depth.
    pub fn get_next_to_calc_at_same_depth(&self) -> Option<Vec<Board>> {
        let mut boards = vec![];
        for (key, value) in &self.scores_player_zero {
            if value.is_none() {
                boards.push(Board::decode(key).unwrap());
            }
        }
        boards.sort_by_key(|x| x.turn);
        let max_turn = boards.iter().map(|x| x.turn).max()?;
        Some(boards.into_iter().filter(|x| x.turn == max_turn).collect())
    }

    pub fn get_unknown_without_unknown_children(&self) -> Option<Vec<Board>> {
        let mut boards = vec![];
        'outer: for (key, value) in &self.scores_player_zero {
            if value.is_some() {
                continue;
            }
            let board = Board::decode(key).unwrap();
            for legal_move in board.next_non_mirrored_moves() {
                let mut next_board = board.clone();
                next_board.game_move(legal_move);
                let entry = self.scores_player_zero.get(&next_board.encode());
                if entry.is_some() && entry.unwrap().is_none() {
                    // has unknown to precalc child
                    continue 'outer;
                }
            }
            boards.push(board)
        }
        if boards.len() >= 1 {
            // Taking higher numbered boards first, gives us a better chance of not getting bottlenecked at the end
            boards.sort_by_key(|x| -(x.turn as i8));
            Some(boards)
        } else {
            None
        }
    }
}

// We want to implement From<MoveStats> for PreCalc
impl From<&MoveStats> for PreCalc {
    fn from(move_stats: &MoveStats) -> Self {
        let board = Board::new();
        let mut precalc = PreCalc {
            scores_player_zero: HashMap::new(),
        };
        // We manually add the mirror board, if desired we can manually add more board we want to precalc
        precalc
            .scores_player_zero
            .insert("12;7E4;7E6;A3h;C3h;E3h;A7h;C7h;E7h".to_string(), None);
        precalc.update_with_move_stats(move_stats, board);
        precalc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn print_move_stats() {
        let move_stats_data: MoveStats =
            serde_json::from_str(&std::fs::read_to_string("move_stats").unwrap()).unwrap();
        move_stats_data.pretty_print(Board::new());
        let mut precalc = PreCalc::from(&move_stats_data);

        let precalc_old = PreCalc::load(
            "./split_up_pre_calcs/:e2:e8:e3:e7:e4:e6:d3h:d7h:e5:f7h:d4h/to_precalc.json",
        )
        .unwrap();
        for (key, value) in precalc_old.scores_player_zero {
            if let Some(value) = value {
                precalc.insert_result(&Board::decode(&key).unwrap(), value);
            }
        }
        println!("{:#?}", precalc.scores_player_zero);
        precalc.store("to_precalc_new.json");
        println!("{:?}", precalc.next_to_calc().map(|x| x.encode()));

        println!(
            "NUMBER OF ENTRIES IS: {:#?}",
            precalc.scores_player_zero.len()
        );
    }
    // Check custom board
    // 12;7E4;7E6;A3h;C3h;E3h;A7h;C7h;E7h

    #[test]
    fn create_custom_move_stats() {
        let to_check_list = [
            "10;8E4;8E6;D3h;F3h;A6h;C6h",
            "11;7E4;8E6;D3h;F3h;H3h;A6h;C6h",
            "12;7E4;8D6;D3h;F3h;H3h;A6h;C6h",
            "12;7E4;7E6;D3h;F3h;H3h;A6h;C6h;E6h",
            "12;7E4;7E6;D3h;F3h;H3h;E4v;A6h;C6h",
            "14;6E4;6E6;D3h;F3h;H3h;E4v;D5v;A6h;C6h;E6h",
        ];
        let mut precalc = PreCalc {
            scores_player_zero: HashMap::new(),
        };
        for board in to_check_list {
            precalc.scores_player_zero.insert(board.to_string(), None);
        }

        precalc.store("precalc_custom.json");
    }
}
