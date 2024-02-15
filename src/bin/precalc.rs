use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use sysinfo::{Process, System};

use quoridor::*;

const PRECALC_FILE: &str = "./to_precalc.json";
const PRECALC_FOLDER: &str = "./precalc";

fn remove_known_moves_for_precalc(mc_ref: &mut MCNode, current_board: &Board, pre_calc: &PreCalc) {
    mc_ref.move_options().unwrap().retain(|game_move| {
        let mut next_board = current_board.clone();
        next_board.game_move(game_move.0);
        pre_calc.roll_out_score(&next_board).is_none()
    })
}

pub fn prune_nodes(mc_node: &mut MCNode, last_visit_cutoff: u32, final_prune: bool) {
    match mc_node {
        MCNode::Branch {
            move_options,
            last_visited,
            scores_included,
            scores,
            ..
        } => {
            if (*last_visited < last_visit_cutoff && scores.1 <= 50_000)
                || (final_prune && scores.1 <= 300_000)
            {
                *move_options = None;
                *scores_included = 0;
            } else {
                if let Some(move_options) = move_options.as_mut() {
                    for (_, node, _) in move_options.iter_mut() {
                        prune_nodes(node, last_visit_cutoff, final_prune);
                    }
                }
            }
        }
        _ => (),
    }
}

fn get_current_process_vms() -> f64 {
    let mut system = System::new_all();
    system.refresh_all();

    let maximum_bytes = 256_000_000_000.0;
    if let Some(process) = system.process(sysinfo::get_current_pid().unwrap()) {
        process.memory() as f64 / maximum_bytes
    } else {
        0.0
    }
}

// Returns whether it has changed
// If the win rate has not changed we can stop updating
fn update_win_rate(board: &Board, precalc: &mut PreCalc) -> Option<bool> {
    let old_win_rate_zero = precalc.roll_out_score(board).map_or(0.0, |x| x);
    let mut ai_board = AIControlledBoard::decode(&board.encode()).unwrap();
    let original_visits = ai_board.relevant_mc_tree.mc_node.number_visits() as f32;
    remove_known_moves_for_precalc(&mut ai_board.relevant_mc_tree.mc_node, board, precalc);
    let visits_after_removal = ai_board.relevant_mc_tree.mc_node.number_visits() as f32;
    println!("VISITS BEFORE REMOVAL: {}", original_visits);
    println!("VISITS AFTER REMOVAL: {}", visits_after_removal);
    if visits_after_removal / original_visits > 0.7 && visits_after_removal > 400_000_000.0 {
        // We don't need to recalculate this node
        let new_score_zero = ai_board.relevant_mc_tree.score_for_zero(board, precalc);
        precalc.insert_result(board, new_score_zero);
        // We will store node with the next step removed.
        ai_board.relevant_mc_tree.serialize_to_file(&format!(
            "{}/{}.mc_node",
            PRECALC_FOLDER,
            board.encode()
        ));

        Some(new_score_zero != old_win_rate_zero)
    } else {
        None
    }
}

// Here we want to return all the plausible boards where the game could end up.
fn next_potential_boards(board: Board) -> Vec<Board> {
    unimplemented!()
}

// After running lots of precalcs for known moves from quoridor games online, Now we want to expand some nodes that are not too deep in the game and where we could end up.
// We will do this seperately for black and white.

fn find_next_board_sequence(ai_player: usize) -> (Vec<Board>, AIControlledBoard) {
    // For the ai player we take the step that the monte carlo algorithm would take online.
    let mut board_sequence = vec![];
    let precalc_file: &str = &"./precalc_full/to_precalc.json";
    let precalc = PreCalc::load(precalc_file).unwrap();
    let mut ai_controlled_board = AIControlledBoard::decode("7;9E4;10E6;D3h").unwrap();

    board_sequence.push(ai_controlled_board.board.clone());
    while precalc
        .roll_out_score(board_sequence.last().unwrap())
        .is_some()
    {
        if ai_controlled_board.board.turn % 2 == ai_player {
            let ai_move = ai_controlled_board.ai_move(0, &precalc);
            ai_controlled_board.game_move(ai_move.0);
        } else {
            // The non ai player takes a random move among the best moves. (for now we say moves that are in the top 10 visit wise
            // and have a win rate that is at most 5% point below the best win rate)
            let ai_move =
                ai_controlled_board.random_good_move(&mut SmallRng::from_entropy(), &precalc);
            ai_controlled_board.game_move(ai_move);
        }
        board_sequence.push(ai_controlled_board.board.clone());
    }
    (board_sequence, ai_controlled_board)
}

fn pre_calculate_board_with_cache(mut known_calc: AIControlledBoard, precalc: &mut PreCalc) {
    if known_calc.relevant_mc_tree.mc_node.number_visits() > 300_000_000 {
        println!(
            "WE Don't need to CALCULATE THIS BOARD: {}, with {} visits",
            known_calc.board.encode(),
            known_calc.relevant_mc_tree.mc_node.number_visits()
        );
        // We will just store this node
        known_calc.relevant_mc_tree.serialize_to_file(&format!(
            "{}/{}.mc_node",
            PRECALC_FOLDER,
            known_calc.board.encode()
        ));
        precalc.insert_result(
            &known_calc.board,
            known_calc
                .relevant_mc_tree
                .score_for_zero(&known_calc.board, precalc),
        );
    } else {
        println!(
            "WE need to CALCULATE THIS BOARD: {}, with {} visits",
            known_calc.board.encode(),
            known_calc.relevant_mc_tree.mc_node.number_visits()
        );

        pre_calculate_board(known_calc.board, precalc);
    }
}

fn pre_calculate_board(board: Board, precalc: &mut PreCalc) {
    let mut ai_board = AIControlledBoard::from_board(board.clone());
    let simulations_per_step = 1_000_000;
    let mut total_simulations = simulations_per_step;
    for _ in 0..(180 * 10) {
        ai_board
            .relevant_mc_tree
            .decide_move(
                board.clone(),
                total_simulations,
                AIControlledBoard::wall_value,
                0.9,
                true,
                true,
                100,
                precalc,
            )
            .unwrap();
        total_simulations += simulations_per_step;
    }
    ai_board.relevant_mc_tree.serialize_to_file(&format!(
        "{}/{}.mc_node",
        PRECALC_FOLDER,
        board.encode()
    ));
    let new_score_zero = ai_board.relevant_mc_tree.score_for_zero(&board, precalc);
    precalc.insert_result(&board, new_score_zero);
}

fn precalc_next_step(ai_player: usize) {
    let (board_sequence, last_board) = find_next_board_sequence(ai_player);

    println!("GOING TO CALCULATE THE FOLLOWING BOARD SEQUENCE");
    for board in &board_sequence {
        println!("{}", board.encode());
    }
    let mut precalc = PreCalc::load(PRECALC_FILE).unwrap();

    pre_calculate_board_with_cache(last_board, &mut precalc);
    precalc.store(PRECALC_FILE);

    for board in board_sequence.into_iter().rev().skip(1) {
        if let Some(changed) = update_win_rate(&board, &mut precalc) {
            if !changed {
                break;
            }
        } else {
            // Now we calculate the board really deeply again;
            pre_calculate_board(board, &mut precalc);
        }
        precalc.store(PRECALC_FILE);
    }
    precalc.store(PRECALC_FILE);
}

//fn main() {
//    let mut to_calc = PreCalc::open(PRECALC_FILE);
//    while let Some(boards) = to_calc.get_unknown_without_unknown_children() {
//        let mut inserts = vec![];
//        std::thread::scope(|s| {
//            {
//                let mut handles = vec![];
//                for board in boards.into_iter().take(8) {
//                    let handle = s.spawn(|| {
//                        println!("PRECALCULATING BOARD: {}", board.encode());
//                        //let board_code = "6;10E4;10E6";
//                        let mut board = AIControlledBoard::from_board(board);
//                        //let relevant_mc_tree =
//                        //    MonteCarloTree::deserialize("14;7E5;7D6;D3h;B4h;D4h;D5v;D7h;F7h.mc_node");
//                        let simulations_per_step = 100_000;
//                        let mut total_simulations = simulations_per_step;
//                        //match &relevant_mc_tree.mc_node {
//                        //    MCNode::Branch { scores, .. } => {
//                        //        total_simulations = scores.1 + simulations_per_step;
//                        //    }
//                        //    _ => (),
//                        //}
//                        //board.relevant_mc_tree = relevant_mc_tree;
//                        let mut i = 1;
//
//                        for _ in 0..(251 * 10) {
//                            // After all legal moves for step one have been expanded.
//                            if i == 3 {
//                                println!(
//                                    "REMOVING KNOWN MOVES: OLD MOVES AMOUNT IS: {}",
//                                    board.relevant_mc_tree.mc_node.move_options().unwrap().len()
//                                );
//                                remove_known_moves_for_precalc(
//                                    &mut board.relevant_mc_tree.mc_node,
//                                    &board.board,
//                                    &to_calc,
//                                );
//                                println!(
//                                    "REMOVING KNOWN MOVES: NEW MOVES AMOUNT IS: {}",
//                                    board.relevant_mc_tree.mc_node.move_options().unwrap().len()
//                                );
//                            }
//                            board
//                                .relevant_mc_tree
//                                .decide_move(
//                                    board.board.clone(),
//                                    total_simulations,
//                                    AIControlledBoard::wall_value,
//                                    0.9,
//                                    true,
//                                    true,
//                                    100,
//                                    &to_calc,
//                                )
//                                .unwrap();
//
//                            total_simulations += simulations_per_step;
//                            println!(
//                                "--------------------- USING {:.4} % BYTES",
//                                get_current_process_vms() * 100.0
//                            );
//                            if get_current_process_vms() > 0.91 {
//                                println!("MEMORY USAGE HAS GOTTEN TOO HIGH, SO WE WILL STOP");
//                                break;
//                            }
//                            // Every Million steps we prune
//                            if i % 25 == 0 {
//                                let start = Instant::now();
//                                let visit_count = board
//                                    .relevant_mc_tree
//                                    .last_visit_count
//                                    .fetch_add(0, Ordering::Relaxed);
//                                let prune_amount = if get_current_process_vms() > 0.85 {
//                                    4_000_000
//                                } else if get_current_process_vms() > 0.7 {
//                                    8_000_000
//                                } else {
//                                    40_000_000
//                                };
//                                if visit_count >= 10_000_000 {
//                                    prune_nodes(
//                                        &mut board.relevant_mc_tree.mc_node,
//                                        visit_count - prune_amount,
//                                        false,
//                                    );
//                                }
//                                println!("Time to prune TREE is: {:?}", start.elapsed());
//                            }
//                            i += 1;
//                        }
//
//                        prune_nodes(
//                            &mut board.relevant_mc_tree.mc_node,
//                            board
//                                .relevant_mc_tree
//                                .last_visit_count
//                                .fetch_add(0, Ordering::Relaxed),
//                            true,
//                        );
//                        std::thread::sleep(std::time::Duration::from_secs(10));
//                        // Here we sleep for a bit, to let the process reclaim its memory.
//
//                        // submoves we are checking
//                        board.relevant_mc_tree.serialize_to_file(&format!(
//                            "{}/{}.mc_node",
//                            PRECALC_FOLDER,
//                            board.board.encode()
//                        ));
//                        (
//                            board.board.clone(),
//                            board
//                                .relevant_mc_tree
//                                .score_for_zero(&board.board, &to_calc),
//                        )
//                    });
//                    handles.push(handle);
//                }
//                for handle in handles {
//                    let (board, win_rate_zero) = handle.join().unwrap();
//                    inserts.push((board, win_rate_zero));
//                }
//            }
//        });
//        for (board, result) in inserts {
//            to_calc.insert_result(&board, result)
//        }
//        to_calc.store(PRECALC_FILE);
//    }
//}

fn main() {
    // We will make this into a loop later on
    precalc_next_step(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_next_board_sequence() {
        // FOR PLAYER 0
        let (board_sequence, last_board) = find_next_board_sequence(0);
        for board in board_sequence {
            println!("{}", board.encode());
        }
        println!(
            "NUMBER OF VISITS LAST BOARD: {}",
            last_board.relevant_mc_tree.mc_node.number_visits()
        );

        // FOR PLAYER 1
        let (board_sequence, last_board) = find_next_board_sequence(1);
        for board in board_sequence {
            println!("{}", board.encode());
        }
        println!(
            "NUMBER OF VISITS LAST BOARD: {}",
            last_board.relevant_mc_tree.mc_node.number_visits()
        );

        precalc_next_step(0);
    }
}
