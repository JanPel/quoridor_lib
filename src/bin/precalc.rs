use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

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

fn main() {
    let board_codes = [
        //"11;8E5;8E6;D3h;D7h;F7h;D4h",
        //"8;9E4;9E6;C3h;D4v",
        "9;9E5;9E6;D3h;D7h",
        "9;8E4;9E6;D3h;C6h;E6v",
        "9;8E4;9E6;D3h;C6h;D5v",
        "9;8E4;9E6;D3h;C6h;F3h",
        "6;10E4;9E7;D4v",
        "7;9E4;10E6;A3h",
        "8;9E4;9E6;A3h;C7h",
        // Mirror board, insert
        "10;8E4;8E6;D3h;F3h;C6h;E6h",
    ];
    let mut to_calc = PreCalc::open(PRECALC_FILE);
    while let Some(boards) = to_calc.get_unknown_without_unknown_children() {
        let mut inserts = vec![];
        std::thread::scope(|s| {
            {
                let mut handles = vec![];
                for board in boards.into_iter().take(8) {
                    let handle = s.spawn(|| {
                        println!("PRECALCULATING BOARD: {}", board.encode());
                        //let board_code = "6;10E4;10E6";
                        let mut board = AIControlledBoard::from_board(board);
                        //let relevant_mc_tree =
                        //    MonteCarloTree::deserialize("14;7E5;7D6;D3h;B4h;D4h;D5v;D7h;F7h.mc_node");
                        let simulations_per_step = 100_000;
                        let mut total_simulations = simulations_per_step;
                        //match &relevant_mc_tree.mc_node {
                        //    MCNode::Branch { scores, .. } => {
                        //        total_simulations = scores.1 + simulations_per_step;
                        //    }
                        //    _ => (),
                        //}
                        //board.relevant_mc_tree = relevant_mc_tree;
                        let mut i = 1;

                        for _ in 0..(251 * 10) {
                            // After all legal moves for step one have been expanded.
                            if i == 3 {
                                println!(
                                    "REMOVING KNOWN MOVES: OLD MOVES AMOUNT IS: {}",
                                    board.relevant_mc_tree.mc_node.move_options().unwrap().len()
                                );
                                remove_known_moves_for_precalc(
                                    &mut board.relevant_mc_tree.mc_node,
                                    &board.board,
                                    &to_calc,
                                );
                                println!(
                                    "REMOVING KNOWN MOVES: NEW MOVES AMOUNT IS: {}",
                                    board.relevant_mc_tree.mc_node.move_options().unwrap().len()
                                );
                            }
                            let _chosen_move = board.ai_move(total_simulations, &to_calc);
                            total_simulations += simulations_per_step;
                            println!(
                                "--------------------- USING {:.4} % BYTES",
                                get_current_process_vms() * 100.0
                            );
                            if get_current_process_vms() > 0.91 {
                                println!("MEMORY USAGE HAS GOTTEN TOO HIGH, SO WE WILL STOP");
                                break;
                            }
                            // Every Million steps we prune
                            if i % 25 == 0 {
                                let start = Instant::now();
                                let visit_count = board
                                    .relevant_mc_tree
                                    .last_visit_count
                                    .fetch_add(0, Ordering::Relaxed);
                                let prune_amount = if get_current_process_vms() > 0.85 {
                                    4_000_000
                                } else if get_current_process_vms() > 0.7 {
                                    8_000_000
                                } else {
                                    40_000_000
                                };
                                if visit_count >= 10_000_000 {
                                    prune_nodes(
                                        &mut board.relevant_mc_tree.mc_node,
                                        visit_count - prune_amount,
                                        false,
                                    );
                                }
                                println!("Time to prune TREE is: {:?}", start.elapsed());
                            }
                            i += 1;
                        }

                        prune_nodes(
                            &mut board.relevant_mc_tree.mc_node,
                            board
                                .relevant_mc_tree
                                .last_visit_count
                                .fetch_add(0, Ordering::Relaxed),
                            true,
                        );
                        std::thread::sleep(std::time::Duration::from_secs(10));
                        // Here we sleep for a bit, to let the process reclaim its memory.

                        // submoves we are checking
                        board.relevant_mc_tree.serialize_to_file(&format!(
                            "{}/{}.mc_node",
                            PRECALC_FOLDER,
                            board.board.encode()
                        ));
                        (
                            board.board.clone(),
                            board
                                .relevant_mc_tree
                                .score_for_zero(&board.board, &to_calc),
                        )
                    });
                    handles.push(handle);
                }
                for handle in handles {
                    let (board, win_rate_zero) = handle.join().unwrap();
                    inserts.push((board, win_rate_zero));
                }
            }
        });
        for (board, result) in inserts {
            to_calc.insert_result(&board, result)
        }
        to_calc.store(PRECALC_FILE);
    }
}
