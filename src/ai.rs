use core::num;
use rand::prelude::IteratorRandom;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::ops::DerefMut;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::board::*;
use crate::move_stats::{self, MoveStats, PreCalc};

const VISIT_LIMIT: u32 = 3_500_000_000;

const PRECALC_FILE: &str = "./split_up_pre_calcs/:e2:e8:e3:e7:e4:e6:d3h:c6h/to_precalc.json";
const PRECALC_FOLDER: &str = "./precalc_full/precalc";

// This is the struct we will use to cache moves, the values are the best move from that board, and how strong the that calculated it was.
pub struct AIControlledBoard {
    pub board: Board,
    #[allow(dead_code)]
    ai_pawn_index: usize,
    pub relevant_mc_tree: MonteCarloTree,
}

const CACHE_SIZE: u32 = 300_000;

#[derive(Clone)]
pub struct CalcCache {
    inner: Arc<Mutex<CalcCacheInner>>,
}

impl CalcCache {
    pub fn zero() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CalcCacheInner::zero())),
        }
    }
    fn get_cache(&mut self, board: &Board, cache_index: u32) -> ([NextMovesCache; 2], u32) {
        if let Some(cache) = self.inner.lock().unwrap().get_inner_cache(cache_index) {
            return (cache, cache_index);
        }
        let new_cache = [NextMovesCache::new(board, 0), NextMovesCache::new(board, 1)];
        let new_index = self.inner.lock().unwrap().insert(new_cache);
        (new_cache, new_index)
    }

    fn insert(&mut self, new_cache: [NextMovesCache; 2]) -> u32 {
        self.inner.lock().unwrap().insert(new_cache)
    }
}

struct CalcCacheInner {
    cache: Vec<[NextMovesCache; 2]>,
    new_cache_index: u32,
}

impl CalcCacheInner {
    pub fn zero() -> Self {
        Self {
            cache: vec![[NextMovesCache::zero(), NextMovesCache::zero()]; CACHE_SIZE as usize],
            new_cache_index: 0,
        }
    }
    fn get_inner_cache(&mut self, cache_index: u32) -> Option<[NextMovesCache; 2]> {
        if self.new_cache_index - cache_index <= CACHE_SIZE {
            let cache = self.cache[(cache_index % CACHE_SIZE) as usize];
            Some(cache)
        } else {
            None
        }
    }

    fn insert(&mut self, new_cache: [NextMovesCache; 2]) -> u32 {
        self.cache[(self.new_cache_index % CACHE_SIZE) as usize] = new_cache;
        self.new_cache_index += 1;
        self.new_cache_index - 1
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum PlayerMoveResult {
    Valid,
    Invalid,
    Win,
    Lose,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub set_up_time: std::time::Duration,
    pub move_option_time: std::time::Duration,
    pub looping_through_move_options_time: std::time::Duration,
    pub ucb_select_time: std::time::Duration,
    pub copy_game_move_time: std::time::Duration,
    pub get_board_score_time: std::time::Duration,
    pub cache_hit_count: usize,
    pub roll_out_time: std::time::Duration,
    pub leaf_select_time: std::time::Duration,
    pub cache_calc_time: std::time::Duration,
    pub new_cache_calc_time: std::time::Duration,
}

impl std::ops::Add<Timings> for Timings {
    type Output = Timings;

    fn add(self, rhs: Timings) -> Self::Output {
        Timings {
            set_up_time: self.set_up_time + rhs.set_up_time,
            move_option_time: self.move_option_time + rhs.move_option_time,
            looping_through_move_options_time: self.looping_through_move_options_time
                + rhs.looping_through_move_options_time,
            ucb_select_time: self.ucb_select_time + rhs.ucb_select_time,
            copy_game_move_time: self.copy_game_move_time + rhs.copy_game_move_time,
            get_board_score_time: self.get_board_score_time + rhs.get_board_score_time,
            cache_hit_count: self.cache_hit_count + rhs.cache_hit_count,
            roll_out_time: self.roll_out_time + rhs.roll_out_time,
            leaf_select_time: self.leaf_select_time + rhs.leaf_select_time,
            cache_calc_time: self.cache_calc_time + rhs.cache_calc_time,
            new_cache_calc_time: self.new_cache_calc_time + rhs.new_cache_calc_time,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MonteCarloTreeSerialize<'a> {
    pub cache_id: u32,
    pub mc_node: &'a MCNode,
    pub last_visit_count: u32,
}

impl<'a> From<&'a MonteCarloTree> for MonteCarloTreeSerialize<'a> {
    fn from(mc_tree: &'a MonteCarloTree) -> Self {
        let last_visit_count = mc_tree.last_visit_count.load(Ordering::Acquire);
        Self {
            cache_id: mc_tree.cache.inner.lock().unwrap().new_cache_index,
            mc_node: &mc_tree.mc_node,
            last_visit_count,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct MonteCarloTreeDeserialize {
    pub cache_id: u32,
    pub mc_node: MCNode,
    pub last_visit_count: u32,
}

impl From<MonteCarloTreeDeserialize> for MonteCarloTree {
    fn from(mc_tree: MonteCarloTreeDeserialize) -> Self {
        let cache = CalcCache::zero();
        cache.inner.lock().unwrap().new_cache_index = mc_tree.cache_id + 2 * CACHE_SIZE;
        // THis is an estimate, we whould probably just serializ the calc cache
        Self {
            cache,
            mc_node: mc_tree.mc_node,
            last_visit_count: Arc::new(AtomicU32::new(mc_tree.last_visit_count)),
        }
    }
}

pub struct MonteCarloTree {
    pub cache: CalcCache,
    pub mc_node: MCNode,
    pub last_visit_count: Arc<AtomicU32>,
}

impl MonteCarloTree {
    fn new() -> Self {
        Self {
            cache: CalcCache::zero(),
            mc_node: MCNode::Leaf,
            last_visit_count: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn serialize_to_file(&self, file_name: &str) {
        let mut file = std::fs::File::create(file_name).unwrap();
        let mut buffer = Vec::new();
        let mc_tree = MonteCarloTreeSerialize::from(self.clone());
        println!(
            "LAST VISIT COUNT IS: {} and cache id = {}",
            mc_tree.last_visit_count, mc_tree.cache_id
        );
        bincode::serialize_into(&mut buffer, &mc_tree).unwrap();
        file.write_all(&buffer).unwrap();
    }

    fn deserialize_file(file_name: &str) -> Self {
        let mut file = std::fs::File::open(file_name).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        Self::deserialize(&buffer)
    }

    pub fn deserialize(buffer: &[u8]) -> Self {
        let deserial_tree: MonteCarloTreeDeserialize = bincode::deserialize(&buffer).unwrap();
        deserial_tree.into()
    }

    // TODO: don't return None
    fn take_move(&mut self, game_move: Move) {
        let mut mc_node = MCNode::Leaf;
        std::mem::swap(&mut self.mc_node, &mut mc_node);
        let mc_node = mc_node.take_move(game_move);
        if let Some(mc_node) = mc_node {
            self.mc_node = mc_node;
        } else {
            *self = Self::new()
        }
    }

    pub fn deeper_known_moves(&self, board: &Board, pre_calc: &PreCalc) -> Vec<(Move, f32)> {
        let mut known_moves = vec![];
        for game_move in board.next_non_mirrored_moves() {
            let mut next_board = board.clone();
            next_board.game_move(game_move);
            if let Some(score_precalc) = pre_calc.roll_out_score(&next_board) {
                let score_for_current_player = if board.turn % 2 == 0 {
                    score_precalc
                } else {
                    1.0 - score_precalc
                };
                known_moves.push((game_move, score_for_current_player));
            }
        }
        known_moves
    }

    fn score_from_deeper_precalc(
        &mut self,
        board: &Board,
        pre_calc: &PreCalc,
    ) -> Option<(Move, f32)> {
        let mut score_deeper: Option<(Move, f32)> = None;
        for (game_move, score_for_current_player) in self.deeper_known_moves(board, pre_calc) {
            if let Some(score_deeper) = &mut score_deeper {
                if score_for_current_player > score_deeper.1 {
                    *score_deeper = (game_move, score_for_current_player);
                }
            } else {
                score_deeper = Some((game_move, score_for_current_player));
            }
        }
        score_deeper
    }

    pub fn score_for_zero(&mut self, board: &Board, pre_calc: &PreCalc) -> f32 {
        let scores = self.mc_node.scores();
        let mut score_current_player = 1.0 - scores.0 / scores.1 as f32;
        println!("SCORE CURRENT PLAYER: {:?}", score_current_player);
        if let Some(score_deeper) = self.score_from_deeper_precalc(board, &pre_calc) {
            if score_deeper.1 > score_current_player {
                score_current_player = score_deeper.1;
            }
        }
        // Bit less fragile then using best move
        if board.turn % 2 == 0 {
            score_current_player
        } else {
            1.0 - score_current_player
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn decide_move(
        &mut self,
        board: Board,
        mut number_of_simulations: u32,
        wall_value: fn(u8) -> f32,
        explore_constant: f32,
        new_logic: bool,
        roll_out_new: bool,
        number_of_averages: u32,
        pre_calc: &PreCalc,
    ) -> Result<(Move, (f32, u32)), Box<dyn std::error::Error + Sync + Send>> {
        number_of_simulations = number_of_simulations.min(3_500_000_000);

        if number_of_simulations >= 1 {
            let mut small_rng = SmallRng::from_entropy();
            let mut timings = Timings::default();
            let start = Instant::now();
            let res = recursive_monte_carlo(
                board.clone(),
                &mut self.mc_node,
                &mut small_rng,
                number_of_simulations,
                0,
                wall_value,
                explore_constant,
                new_logic,
                &mut timings,
                None,
                roll_out_new,
                number_of_averages,
                &mut self.cache,
                &self.last_visit_count.clone(),
                &pre_calc,
            );
            log::info!("Deciding On move took: {:?}", start.elapsed());
        }
        //println!("{:?}", timings);
        let move_options = self
            .mc_node
            .move_options()
            .ok_or("Need to enough simulations to expand root leaf")?;
        move_options.sort_by_key(|x| x.1.number_visits());
        let best_move =
            select_robust_best_branch(move_options, &board).ok_or("Best move is unclear")?;
        if let Some(precalc_option) = self.score_from_deeper_precalc(&board, &pre_calc) {
            log::info!("best precalc alernativie is {:?}", precalc_option);
            if precalc_option.1 >= (best_move.1 .0 / best_move.1 .1 as f32) * 0.98 {
                log::info!("best precalc alernativie better");
                return Ok((
                    precalc_option.0,
                    (precalc_option.1 * 250_000_000.0, 250_000_000),
                ));
            }
        }
        Ok(best_move)
    }
    // We only want to include this function with non wasm
    #[cfg(not(target_arch = "wasm32"))]
    pub fn decide_move(
        &mut self,
        board: Board,
        mut number_of_simulations: u32,
        wall_value: fn(u8) -> f32,
        explore_constant: f32,
        new_logic: bool,
        roll_out_new: bool,
        number_of_averages: u32,
        pre_calc: &PreCalc,
    ) -> Result<(Move, (f32, u32)), Box<dyn std::error::Error + Sync + Send>> {
        if number_of_simulations >= self.mc_node.number_visits() {
            if self.mc_node.number_visits() >= VISIT_LIMIT {
                // We're running too close to the integer limit
                number_of_simulations = 0;
            } else {
                number_of_simulations -= self.mc_node.number_visits();
            }
        } else {
            number_of_simulations = 0;
        }

        if number_of_simulations >= 1 {
            let mut small_rng = SmallRng::from_entropy();
            let mut timings = Timings::default();
            let start = Instant::now();
            let res = multithreaded_mc(
                board.clone(),
                &mut self.mc_node,
                10,
                number_of_simulations,
                &mut small_rng,
                0,
                wall_value,
                explore_constant,
                new_logic,
                &mut timings,
                roll_out_new,
                number_of_averages,
                &mut self.cache,
                self.last_visit_count.clone(),
                &pre_calc,
            );
            if res.last_visit_count >= 10_000_000 {
                let start = Instant::now();

                println!("Pruning nodes, took {:?}", start.elapsed());
            }
            println!("Deciding On move took: {:?}", start.elapsed());
        }
        //println!("{:?}", timings);
        let move_options = self
            .mc_node
            .move_options()
            .ok_or("Need to enough simulations to expand root leaf")?;
        move_options.sort_by_key(|x| x.1.number_visits());
        let best_move =
            select_robust_best_branch(move_options, &board).ok_or("Best move is unclear")?;
        if let Some(precalc_option) = self.score_from_deeper_precalc(&board, &pre_calc) {
            println!("best precalc alernativie is {:?}", precalc_option);
            if precalc_option.1 > best_move.1 .0 / best_move.1 .1 as f32 {
                println!("best precalc alernativie better");
                return Ok((
                    precalc_option.0,
                    (precalc_option.1 * 250_000_000.0, 250_000_000),
                ));
            }
        }
        Ok(best_move)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MCNode {
    Leaf,
    Branch {
        move_options: Option<Box<Vec<(Move, MCNode, i8)>>>,
        scores_included: i8,
        scores: (f32, u32),
        cache_index: u32,
        last_visited: u32,
    },
    PlayedOut {
        scores: (f32, u32),
        move_options: Option<Box<Vec<(Move, MCNode, i8)>>>,
    },
}

impl MCNode {
    fn take_move(self, game_move: Move) -> Option<Self> {
        match self {
            MCNode::Leaf => None,
            MCNode::Branch { move_options, .. } => {
                for move_option in move_options?.into_iter() {
                    if move_option.0 == game_move {
                        return Some(move_option.1);
                    }
                }
                None
            }
            MCNode::PlayedOut { move_options, .. } => {
                for move_option in move_options?.into_iter() {
                    if move_option.0 == game_move {
                        return Some(move_option.1);
                    }
                }
                None
            }
        }
    }
    fn is_leaf(&self) -> bool {
        match self {
            MCNode::Leaf => true,
            MCNode::Branch { .. } => false,
            MCNode::PlayedOut { .. } => false,
        }
    }
    fn played_out(&self) -> bool {
        match self {
            MCNode::Leaf => false,
            MCNode::Branch { .. } => self.number_visits() >= VISIT_LIMIT,
            MCNode::PlayedOut { .. } => true,
        }
    }

    pub fn number_visits(&self) -> u32 {
        match self {
            MCNode::Leaf => 0,
            MCNode::Branch { scores, .. } => scores.1,
            MCNode::PlayedOut { scores, .. } => scores.1,
        }
    }

    pub fn move_options(&mut self) -> Option<&mut Vec<(Move, MCNode, i8)>> {
        match self {
            MCNode::Branch { move_options, .. } => move_options.as_mut().map(|x| x.deref_mut()),
            MCNode::PlayedOut { move_options, .. } => move_options.as_mut().map(|x| x.deref_mut()),
            _ => None,
        }
    }

    fn scores(&self) -> (f32, u32) {
        match self {
            MCNode::Leaf => (1.0, 1),
            MCNode::Branch { scores, .. } => *scores,
            MCNode::PlayedOut { scores, .. } => *scores,
        }
    }

    fn ucb_score(&self, heuristic_score: i8, parent_visits: u32, explore_constant: f32) -> f32 {
        match self {
            MCNode::Leaf => 3.0 + heuristic_score as f32,
            MCNode::Branch { scores, .. } => ucb_score(
                scores.0,
                scores.1,
                parent_visits,
                explore_constant,
                heuristic_score,
            ),
            MCNode::PlayedOut { scores, .. } => scores.0 / scores.1 as f32,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn multithreaded_mc(
    board: Board,
    mc_ref: &mut MCNode,
    number_threads: u32,
    number_of_loops: u32,
    small_rng: &mut SmallRng,
    depth: u32,
    wall_value: fn(u8) -> f32,
    explore_constant: f32,
    new_logic: bool,
    timings: &mut Timings,
    roll_out_new: bool,
    number_of_averages: u32,
    calc_cache: &mut CalcCache,
    last_visit_count: Arc<AtomicU32>,
    precalc: &PreCalc,
) -> MonteCarloResponse {
    let number_of_loops_first =
        if number_threads == 1 || mc_ref.played_out() || number_of_loops <= 100 {
            number_of_loops
        } else {
            3
        };
    let total_res = recursive_monte_carlo(
        board.clone(),
        mc_ref,
        small_rng,
        number_of_loops_first,
        depth,
        wall_value,
        explore_constant,
        new_logic,
        timings,
        None,
        roll_out_new,
        number_of_averages,
        calc_cache,
        &last_visit_count.clone(),
        precalc,
    );

    if number_threads == 1 || mc_ref.played_out() || number_of_loops <= 100 {
        return total_res;
    }

    let moves = mc_ref.move_options().unwrap();
    let total_res = std::thread::scope(|s| {
        let mut handles = vec![];
        let (split_moves, best_option) = split_up_move_options(
            moves,
            number_threads,
            number_of_loops / number_threads,
            explore_constant,
        );
        if let Some((best_move, best_mc, number_of_threads_to_best)) = best_option {
            let mut next_board = board.clone();
            next_board.game_move(best_move);
            let number_of_loops = number_of_loops / number_threads * number_of_threads_to_best;
            //println!(
            //    "BEST MOVE: {:?}, number of simulations: {}",
            //    best_move, number_of_loops
            //);
            let mut calc_cache = calc_cache.clone();
            let last_visit_count = last_visit_count.clone();
            handles.push(s.spawn(move || {
                multithreaded_mc(
                    next_board,
                    best_mc,
                    number_of_threads_to_best,
                    number_of_loops,
                    small_rng,
                    depth + 1,
                    wall_value,
                    explore_constant,
                    new_logic,
                    timings,
                    roll_out_new,
                    number_of_averages,
                    &mut calc_cache,
                    last_visit_count,
                    precalc,
                )
            }));
        };
        let mut total_res = total_res;
        for moves in split_moves {
            let board = board.clone();
            let mut calc_cache = calc_cache.clone();
            let last_visit_count = last_visit_count.clone();
            let handle = s.spawn(move || {
                monte_carlo_moves(
                    board.clone(),
                    number_of_loops / number_threads as u32,
                    moves,
                    &mut SmallRng::from_entropy(),
                    depth + 1,
                    wall_value,
                    explore_constant,
                    new_logic,
                    &mut Timings::default(),
                    roll_out_new,
                    number_of_averages,
                    &mut calc_cache,
                    last_visit_count,
                    precalc,
                ) // test
            });
            handles.push(handle);
        }
        for handle in handles {
            let res = handle.join().unwrap();
            total_res.game_count += res.game_count;
            total_res.score_zero += res.score_zero;
            total_res.last_visit_count = total_res.last_visit_count.max(res.last_visit_count);
        }
        total_res
    });
    match mc_ref {
        MCNode::Branch {
            scores,
            last_visited,
            ..
        } => {
            let prev_board = (board.turn + 1) % 2;

            *last_visited = total_res.last_visit_count;
            scores.1 += total_res.game_count;
            if prev_board == 0 {
                scores.0 += total_res.score_zero;
            } else {
                scores.0 += total_res.game_count as f32 - total_res.score_zero;
            }
        }
        _ => (),
    };
    total_res
}

#[cfg(not(target_arch = "wasm32"))]
fn split_up_move_options(
    move_options: &mut [(Move, MCNode, i8)],
    mut number_of_threads: u32,
    simulations_per_thread: u32,
    explore_constant: f32,
) -> (
    Vec<&mut [(Move, MCNode, i8)]>,
    Option<(Move, &mut MCNode, u32)>,
) {
    let parent_visits = move_options.iter().map(|x| x.1.number_visits()).sum();

    move_options.sort_by_key(|x| {
        -(x.1.ucb_score(x.2.max(0), parent_visits, explore_constant) * 1000000.0) as i32
    });
    let mut number_threads_to_best = {
        let (game_move, _, number_of_next_to_best, _) =
            select_next_node(move_options, parent_visits, explore_constant).unwrap();
        //println!(
        //    "Number of next to best: {}, move is: {:?}",
        //    number_of_next_to_best, game_move
        //);
        (number_of_next_to_best / simulations_per_thread).min(number_of_threads)
    };

    //println!("number of threads to best: {}", number_threads_to_best);

    let mut moves_cutoff_index = move_options.len();

    let scores_best: (f32, u32) = move_options[0].1.scores();
    let best_heuristic = move_options[0].2;
    for (index, move_option) in move_options.iter().enumerate().skip(1) {
        let treshold_y =
            move_option
                .1
                .ucb_score(move_option.2.max(0), parent_visits, explore_constant);
        let losses_estimated = estimate_losses(
            scores_best.0,
            scores_best.1,
            parent_visits,
            explore_constant,
            treshold_y,
            best_heuristic,
        );
        if losses_estimated >= simulations_per_thread * number_of_threads {
            moves_cutoff_index = index;
            break;
        }
    }
    let move_options = &mut move_options[0..moves_cutoff_index];
    // We have more threads then options we want to explore in this step. (cause the rest of the options are too bad)

    number_of_threads -= number_threads_to_best;
    if move_options.len() as u32 - 1 <= number_of_threads {
        let diff = number_of_threads - (move_options.len() - 1) as u32;
        number_threads_to_best += diff;
        number_of_threads -= diff;
    }

    let mut remainder;
    let mut best_move_split = None;
    if number_threads_to_best >= 1 {
        let (split_1, split_2) = move_options.split_at_mut(1);
        let best_move = split_1.last_mut().unwrap();
        best_move_split = Some((best_move.0, &mut best_move.1, number_threads_to_best));
        remainder = split_2;
    } else {
        remainder = move_options;
    }

    let mut result = vec![];
    if number_of_threads >= 1 {
        if number_of_threads >= 2 {
            let mut splits = vec![0; number_of_threads as usize];
            let mut index = 0;
            let mut index_add = 1;
            remainder.sort_by_cached_key(|x| {
                let key: i32 = index % number_of_threads as i32;
                index += index_add;
                if index % number_of_threads as i32 == 0 {
                    index_add *= -1;
                }
                splits[key as usize] += 1;
                key
            });

            // Here we will first just split it equally.
            for &index in &splits[0..(splits.len() - 1)] {
                if remainder.len() <= index as usize {
                    break;
                } else if index == 0 {
                    continue;
                }
                let (split_1, split_2) = remainder.split_at_mut(index);
                result.push(split_1);
                remainder = split_2;
            }
            if remainder.len() >= 1 {
                result.push(remainder);
            }
        } else {
            if remainder.len() >= 1 {
                result.push(remainder);
            }
        }
    }
    (result, best_move_split)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn monte_carlo_moves(
    board: Board,
    number_of_loops: u32,
    move_options: &mut [(Move, MCNode, i8)],
    small_rng: &mut SmallRng,
    depth: u32,
    wall_value: fn(u8) -> f32,
    explore_constant: f32,
    new_logic: bool,
    timings: &mut Timings,
    roll_out_new: bool,
    number_of_averages: u32,
    calc_cache: &mut CalcCache,
    last_visit_count: Arc<AtomicU32>,
    pre_calc_scores: &PreCalc,
) -> MonteCarloResponse {
    let mut to_return = MonteCarloResponse::new();
    let mut parent_visits = move_options.iter().map(|x| x.1.number_visits()).sum();

    while to_return.game_count < number_of_loops {
        let result = {
            // If there are leaves, we select one, run through it and continue.
            // If there are no leaves, we select the most promising branch.
            let (chosen_move, node, moves_allowed, played_out_node) =
                select_next_node(move_options, parent_visits, explore_constant)
                    .expect("Next moves is empty");

            let mut next_board = board.clone();
            next_board.game_move(*chosen_move);

            let result = recursive_monte_carlo(
                next_board,
                node,
                small_rng,
                moves_allowed.min(number_of_loops - to_return.game_count),
                depth + 1,
                wall_value,
                explore_constant,
                new_logic,
                timings,
                None,
                roll_out_new,
                number_of_averages,
                calc_cache,
                &last_visit_count,
                pre_calc_scores,
            );
            result
        };
        to_return.score_zero += result.score_zero;
        to_return.game_count += result.game_count;
        to_return.last_visit_count = result.last_visit_count;

        // Now we will aslo update the scores
        parent_visits += result.game_count;
    }
    to_return
}

#[derive(Clone, Copy, Debug)]
pub struct MonteCarloResponse {
    pub score_zero: f32,
    pub game_count: u32,
    pub last_visit_count: u32,
}

impl MonteCarloResponse {
    fn new() -> Self {
        Self {
            score_zero: 0.0,
            game_count: 0,
            last_visit_count: 0,
        }
    }
}

pub fn recursive_monte_carlo(
    board: Board,
    mc_ref: &mut MCNode,
    small_rng: &mut SmallRng,
    number_of_loops: u32,
    depth: u32,
    wall_value: fn(u8) -> f32,
    explore_constant: f32,
    new_logic: bool,
    timings: &mut Timings,
    previous_move_cache: Option<[NextMovesCache; 2]>,
    roll_out_new: bool,
    number_of_averages: u32,
    calc_cache: &mut CalcCache,
    last_visit_count: &Arc<AtomicU32>,
    precalc: &PreCalc,
) -> MonteCarloResponse {
    let mut to_return = MonteCarloResponse::new();
    //let mut score_zero = 0.0;
    //let mut game_count: u32 = 0;
    //let mut to_return_visit_count = 0;

    let prev_board = (board.turn + 1) % 2;
    let mut played_out: Option<(f32, Option<usize>)> = None;
    while to_return.game_count < number_of_loops {
        if depth == 0 && (to_return.game_count + 1) % 100_000 == 0 {
            println!("game count: {}", to_return.game_count);
        }
        match mc_ref {
            MCNode::Leaf => {
                //let start = Instant::now();
                //let next_cache = [
                //    NextMovesCache::new(&board, 0),
                //    NextMovesCache::new(&board, 1),
                //];
                //timings.new_cache_calc_time += start.elapsed();

                let next_moves_cache = if let Some(previous_move_cache) = previous_move_cache {
                    //for i in 0..2 {
                    //    assert_eq!(
                    //        previous_move_cache[i].distances_to_finish,
                    //        next_cache[i].distances_to_finish,
                    //    );
                    //    if previous_move_cache[i].allowed_walls_for_pawn
                    //        != next_cache[i].allowed_walls_for_pawn
                    //    {
                    //        println!("{}", board.encode());
                    //        println!(
                    //            "allowed walls for pawn {}: {:?} are different",
                    //            i, board.pawns[i]
                    //        );
                    //        println!("allowed walls previous_cache");
                    //        previous_move_cache[i]
                    //            .allowed_walls_for_pawn
                    //            .pretty_print_wall();

                    //        println!("{}", board.encode());
                    //        println!("allowed walls new cache");
                    //        next_cache[i].allowed_walls_for_pawn.pretty_print_wall();

                    //        println!("walls");
                    //        board.walls.pretty_print_wall();

                    //        println!("{:#?}", previous_move_cache[i].allowed_walls_for_pawn);
                    //        println!("{:#?}", next_cache[i].allowed_walls_for_pawn);
                    //        println!("{:#?}", next_cache[i].allowed_walls_for_pawn);
                    //    }
                    //    assert_eq!(
                    //        previous_move_cache[i].allowed_walls_for_pawn,
                    //        next_cache[i].allowed_walls_for_pawn,
                    //    );
                    //    if previous_move_cache[i].relevant_squares != next_cache[i].relevant_squares
                    //    {
                    //        println!("{}", board.encode());
                    //        println!(
                    //            "allowed walls for pawn {}: {:?} are different",
                    //            i, board.pawns[i]
                    //        );
                    //        println!("{}", board.encode());
                    //        println!("previous_cache");
                    //        previous_move_cache[i].relevant_squares.pretty_print();
                    //        println!("new cache");
                    //        next_cache[i].relevant_squares.pretty_print();

                    //        println!("walls");
                    //        board.walls.pretty_print_wall();
                    //    }
                    //    assert_eq!(
                    //        previous_move_cache[i].relevant_squares,
                    //        next_cache[i].relevant_squares,
                    //    );
                    //}

                    previous_move_cache
                } else {
                    let next_cache = [
                        NextMovesCache::new(&board, 0),
                        NextMovesCache::new(&board, 1),
                    ];
                    next_cache
                };

                // In case we hit a precalculated node
                // We probably want to include an enum type for this scenario....
                // Hmm of course this is not the same as rolled out.
                if depth >= 1 {
                    if let Some(roll_out_result) = precalc.roll_out_score(&board) {
                        to_return.game_count += 1;
                        to_return.score_zero += roll_out_result;
                        to_return.last_visit_count =
                            last_visit_count.fetch_add(1, Ordering::AcqRel);
                        let board_score = if board.turn % 2 != 0 {
                            to_return.score_zero
                        } else {
                            1.0 - to_return.score_zero
                        };
                        *mc_ref = MCNode::PlayedOut {
                            scores: (board_score, 1),
                            move_options: None,
                        };
                        continue;
                    }
                }
                let (roll_out_score, game_finish_leave) = board.roll_out(
                    small_rng,
                    wall_value,
                    &next_moves_cache,
                    roll_out_new,
                    number_of_averages,
                );
                //board.roll_out_two_steps_greedy(small_rng)
                to_return.game_count += 1;
                to_return.score_zero += roll_out_score;
                to_return.last_visit_count = last_visit_count.fetch_add(1, Ordering::AcqRel);
                // This is wrong, now it will keep picking losing moves.
                let board_score = if board.turn % 2 != 0 {
                    to_return.score_zero
                } else {
                    1.0 - to_return.score_zero
                };
                if game_finish_leave && depth >= 1 {
                    *mc_ref = MCNode::PlayedOut {
                        scores: (board_score, 1),
                        move_options: None,
                    };
                } else {
                    let cache_index = calc_cache.insert(next_moves_cache);
                    *mc_ref = MCNode::Branch {
                        move_options: None,
                        scores_included: 0,
                        scores: (board_score, 1),
                        cache_index,
                        last_visited: to_return.last_visit_count,
                    };
                }
            }
            MCNode::PlayedOut { scores, .. } => {
                let score = scores.0 / scores.1 as f32;
                let result = if prev_board == 0 { score } else { 1.0 - score };
                let multiplier = number_of_loops - to_return.game_count;
                scores.0 += score * multiplier as f32;
                scores.1 += multiplier;
                to_return.score_zero += result * multiplier as f32;
                to_return.game_count += multiplier;
                to_return.last_visit_count = last_visit_count.fetch_add(1, Ordering::AcqRel);
            }
            MCNode::Branch {
                move_options,
                scores,
                scores_included,
                cache_index,
                last_visited,
            } => {
                if let Some(&played_out) = played_out.as_ref() {
                    // Try removing  + 1 later on
                    let scores = (played_out.0 * (scores.1 + 1) as f32, (scores.1 + 1));
                    let mut new_moves_options = vec![];
                    std::mem::swap(&mut new_moves_options, move_options.as_mut().unwrap());
                    if let Some(played_index) = played_out.1 {
                        new_moves_options = vec![new_moves_options.swap_remove(played_index)];
                    }
                    *mc_ref = MCNode::PlayedOut {
                        scores,
                        move_options: Some(Box::new(new_moves_options)),
                    };
                    continue;
                }
                // If we haven't filled move options yet, we will do so first (this is an optimization)
                if move_options.is_none() {
                    let (next_moves_cache, new_cache_index) =
                        calc_cache.get_cache(&board, *cache_index);
                    *cache_index = new_cache_index;
                    let mut next_moves: Vec<(Move, MCNode, i8)> = board
                        .next_moves_with_scoring(true, small_rng, &next_moves_cache)
                        .into_iter()
                        .map(|x| (x.0, MCNode::Leaf, x.1))
                        .collect();

                    // This is just a sanity check for some bugs we are seeing, can be removed later on again.
                    if next_moves.len() == 0 {
                        panic!("No next moves for board {}", board.encode());
                    }
                    let number_of_positive_scores = next_moves.iter().filter(|x| x.2 >= 0).count();
                    if number_of_positive_scores >= 1 {
                        next_moves = next_moves.into_iter().filter(|x| x.2 >= 0).collect();
                        // To make sure we don't waste space.
                    }
                    next_moves.shrink_to_fit();
                    *move_options = Some(Box::new(next_moves));
                }
                // If we have visited a node 10_000 times, we want to include all other options to consider
                if scores.1 >= 10_000 && *scores_included == 0 {
                    let (next_moves_cache, new_cache_index) =
                        calc_cache.get_cache(&board, *cache_index);
                    *cache_index = new_cache_index;
                    let next_moves =
                        &board.next_moves_with_scoring(true, small_rng, &next_moves_cache)
                            [move_options.as_ref().unwrap().len()..];
                    move_options
                        .as_mut()
                        .unwrap()
                        .extend(next_moves.iter().map(|x| (x.0, MCNode::Leaf, x.1)));
                    move_options.as_mut().map(|x| x.shrink_to_fit());

                    // This is just a sanity check for some bugs we are seeing, can be removed later on again.
                    if move_options.as_ref().map_or(0, |x| x.len()) == 0 {
                        panic!("No next moves for board {}", board.encode());
                    }

                    *scores_included = -1;
                }

                let result = {
                    // If there are leaves, we select one, run through it and continue.
                    // If there are no leaves, we select the most promising branch.
                    let number_move_options = move_options.as_ref().map_or(0, |x| x.len());
                    let (chosen_move, node, moves_allowed, played_out_node) = select_next_node(
                        move_options.as_mut().unwrap(),
                        scores.1,
                        explore_constant,
                    )
                    .unwrap_or_else(|| {
                        panic!(
                            "Next moves is empty for board {}, move options len is {}",
                            board.encode(),
                            number_move_options
                        )
                    });
                    if let Some(played_out_score) = played_out_node {
                        // In this case we want to also check the unlikely options, to be really sure this board is played out.
                        if played_out_score.0 < 0.5 && *scores_included == 0 {
                            let (next_moves_cache, new_cache_index) =
                                calc_cache.get_cache(&board, *cache_index);
                            *cache_index = new_cache_index;
                            let next_moves =
                                &board.next_moves_with_scoring(true, small_rng, &next_moves_cache)
                                    [move_options.as_ref().unwrap().len()..];
                            move_options
                                .as_mut()
                                .unwrap()
                                .extend(next_moves.iter().map(|x| (x.0, MCNode::Leaf, x.1)));
                            move_options.as_mut().map(|x| x.shrink_to_fit());

                            // This is just a sanity check for some bugs we are seeing, can be removed later on again.
                            if move_options.as_ref().map_or(0, |x| x.len()) == 0 {
                                panic!("No next moves for board {}", board.encode());
                            }

                            *scores_included = -1;
                            continue;
                        }
                        played_out = Some(played_out_score);
                        continue;
                    }

                    let mut next_board = board.clone();
                    next_board.game_move(*chosen_move);

                    let previous_move_cache = if node.is_leaf() {
                        //println!(
                        //"previous board: {}, next_board: {}, game_move: {:?}",
                        //board.encode(),
                        //next_board.encode(),
                        //chosen_move
                        //);
                        let (cache, new_cache_index) = calc_cache.get_cache(&board, *cache_index);
                        *cache_index = new_cache_index;
                        let res = [
                            cache[0].next_cache(*chosen_move, &board, &next_board, 0),
                            cache[1].next_cache(*chosen_move, &board, &next_board, 1),
                        ];
                        timings.cache_hit_count += 1;
                        Some(res)
                    } else {
                        None
                    };
                    let result = recursive_monte_carlo(
                        next_board,
                        node,
                        small_rng,
                        moves_allowed.min(number_of_loops - to_return.game_count),
                        depth + 1,
                        wall_value,
                        explore_constant,
                        new_logic,
                        timings,
                        previous_move_cache,
                        roll_out_new,
                        number_of_averages,
                        calc_cache,
                        last_visit_count,
                        precalc,
                    );
                    result
                };
                to_return.score_zero += result.score_zero;
                to_return.game_count += result.game_count;
                to_return.last_visit_count = result.last_visit_count;

                // Now we will aslo update the scores
                scores.1 += result.game_count;
                if prev_board == 0 {
                    scores.0 += result.score_zero;
                } else {
                    scores.0 += result.game_count as f32 - result.score_zero;
                }
                *last_visited = result.last_visit_count;
            }
        }
    }

    to_return
}

impl AIControlledBoard {
    pub fn is_played_out(&self) -> bool {
        self.relevant_mc_tree.mc_node.played_out()
    }
    pub fn player_first() -> Self {
        // we want to deserialize the move stats from the file move_stats
        Self {
            board: Board::new(),
            ai_pawn_index: 1,
            relevant_mc_tree: MonteCarloTree::new(),
        }
    }

    pub fn decode(encoded: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let board = Board::decode(encoded)?;

        let mut relevant_mc_tree = MonteCarloTree::new();
        if cfg!(not(target_arch = "wasm32")) {
            // if in the folder precalc we have the board, we want to load the precalculated data into the MontecarloTree
            let board_path = format!("{}/{}.mc_node", PRECALC_FOLDER, encoded);
            if std::path::Path::new(&board_path).exists() {
                relevant_mc_tree = MonteCarloTree::deserialize_file(&board_path);
            }
        }

        Ok(Self {
            board,
            ai_pawn_index: 0,
            relevant_mc_tree,
        })
    }
    pub fn from_board(board: Board) -> Self {
        Self {
            board,
            ai_pawn_index: 0,
            relevant_mc_tree: MonteCarloTree::new(),
        }
    }

    pub fn game_move(&mut self, game_move: Move) -> (MoveResult, u32) {
        let move_result = self.board.game_move(game_move);
        if move_result == MoveResult::Invalid {
            return (move_result, 0);
        }
        println!("{}", self.board.encode());
        let encoded = self.board.encode();
        assert_eq!(
            Board::decode(&encoded).unwrap().minimal_board(),
            self.board.minimal_board()
        );
        let mut hit_branch = false;
        let mut number_of_simulations = 0;
        self.relevant_mc_tree.take_move(game_move);
        match &self.relevant_mc_tree.mc_node {
            MCNode::Branch { scores, .. } => {
                hit_branch = true;
                println!(
                    "player chose move which we gave a win chance of {:.1} and total visits {}",
                    scores.0 / scores.1 as f32 * 100.0,
                    scores.1
                );
                number_of_simulations = scores.1;
            }
            MCNode::Leaf => {
                println!("opponent chose move we didn't consider");
            }
            MCNode::PlayedOut {
                scores,
                move_options,
            } => {
                if let Some(move_options) = move_options.as_ref() {
                    if move_options.len() >= 1 {
                        hit_branch = true;
                    }
                }
                // For now we don't want to continue with played out moves, cause we don't know what move to do..
                println!(
                    "opponent choose move of a played out branch with win chance {:.} ",
                    scores.0 / scores.1 as f32 * 100.0
                );
            }
        }
        if !hit_branch {
            self.relevant_mc_tree = MonteCarloTree::new();
        }
        // if in the folder precalc we have the board, we want to load the precalculated data into the MontecarloTree
        if cfg!(not(target_arch = "wasm32")) {
            // if in the folder precalc we have the board, we want to load the precalculated data into the MontecarloTree
            let board_path = format!("{}/{}.mc_node", PRECALC_FOLDER, encoded);
            if std::path::Path::new(&board_path).exists() {
                self.relevant_mc_tree = MonteCarloTree::deserialize_file(&board_path);
                number_of_simulations = self.relevant_mc_tree.mc_node.number_visits();
            }
        }

        (move_result, number_of_simulations)
    }

    pub fn ai_move(
        &mut self,
        number_of_steps: u32,
        pre_calc: &PreCalc,
    ) -> (Move, (usize, usize), u32) {
        let ai_move = self
            .relevant_mc_tree
            .decide_move(
                self.board.clone(),
                number_of_steps,
                Self::wall_value,
                0.5,
                true,
                true,
                100,
                pre_calc,
            )
            .unwrap();
        log::info!(
            "AI move estimates {:?}, so win rate is {:.1}%",
            (ai_move.0, ai_move.1),
            ai_move.1 .0 as f32 / ai_move.1 .1 as f32 * 100.0
        );
        //};
        match ai_move.0 {
            Move::PawnMove(first_step, second_step) => {
                let mut next_board = self.board.clone();
                next_board.game_move(Move::PawnMove(first_step, second_step));
                let pos = next_board.pawns[self.board.turn % 2].position;

                (
                    Move::PawnMove(first_step, second_step),
                    (pos.row as usize, pos.col as usize),
                    ai_move.1 .1,
                )
            }
            Move::Wall(dir, loc) => (
                Move::Wall(dir, loc),
                (loc.row as usize, loc.col as usize),
                ai_move.1 .1,
            ),
        }
    }

    pub fn random_good_move(&mut self, small_rng: &mut SmallRng, pre_calc: &PreCalc) -> Move {
        let deeper_good_moves = self
            .relevant_mc_tree
            .deeper_known_moves(&self.board, pre_calc);

        let move_options = self.relevant_mc_tree.mc_node.move_options().unwrap();

        let mut combined_moves = vec![];

        for (game_move, mc_node, _) in move_options.iter() {
            if let Some((_, win_rate)) = deeper_good_moves.iter().find(|x| x.0 == *game_move) {
                combined_moves.push((*game_move, *win_rate));
            } else {
                combined_moves.push((*game_move, mc_node.scores().0 / mc_node.scores().1 as f32));
            }
        }

        for (game_move, win_rate) in deeper_good_moves {
            if combined_moves.iter().find(|x| x.0 == game_move).is_some() {
                continue;
            }
            combined_moves.push((game_move, win_rate));
        }
        combined_moves.sort_by_key(|x| (-x.1 * 100000.0) as i32);

        let max_win_rate = combined_moves[0].1;
        let to_select: Vec<_> = combined_moves
            .into_iter()
            .filter(|x| x.1 > max_win_rate * 0.95)
            .take(10)
            .collect();

        println!("good move options: {:?}", to_select);
        // Now we want to take a random element form the iterator
        to_select.into_iter().choose(small_rng).unwrap().0
    }

    pub fn wall_value(input: u8) -> f32 {
        if input <= 1 {
            // the last wall has a lot of value, cause playing it gives the opponent free reign
            4.0 * input as f32
        } else if input <= 5 {
            // the next walls have wall value 3.
            (4 + 3 * (input - 1)) as f32
        } else if input <= 8 {
            // The next walls we give value 2
            (4 + 12 + 2 * (input - 5)) as f32
        } else {
            // TEMP value
            4.0 + 12.0 + 6.0
            // first two walls we give value 1, cause hard to use in effective way
            //(4 + 12 + 8 + input - 8) as f32
        }
    }

    // this function return the (number of wins for player 0, number of games played)
}

impl Board {
    pub fn roll_out(
        &self,
        small_rng: &mut SmallRng,
        wall_value: fn(u8) -> f32,
        cache: &[NextMovesCache; 2],
        roll_out_new: bool,
        number_of_averages: u32,
    ) -> (f32, bool) {
        //println!(
        //    "{:?}: squares_left: {}, dist {}, {:?}, squares left: {}, dist {}",
        //    self.pawns[0],
        //    cache[0].relevant_squares.number_of_squares,
        //    cache[0].distances_to_finish.dist[self.pawns[0].position].unwrap() as f32,
        //    self.pawns[1],
        //    cache[1].relevant_squares.number_of_squares,
        //    cache[1].distances_to_finish.dist[self.pawns[1].position].unwrap() as f32,
        //);
        if let Some(score) = self.roll_out_finish(cache) {
            let k = 1.0;
            let score = 1.0 / (1.0 + (-k * score).exp());

            return (score, true);
        }

        let max_length = if roll_out_new
            && self.pawns[0].number_of_walls_left >= 4
            && self.pawns[1].number_of_walls_left >= 4
        {
            let blocked_off_0 = self.open_routes.test_check_lines(
                self.pawns[0],
                &cache[0].relevant_squares,
                cache[0].distances_to_finish,
            );
            let blocked_off_1 = self.open_routes.test_check_lines(
                self.pawns[1],
                &cache[1].relevant_squares,
                cache[1].distances_to_finish,
            );

            [
                (cache[0]
                    .relevant_squares
                    .number_of_left_relevant(blocked_off_0) as f32),
                (cache[1]
                    .relevant_squares
                    .number_of_left_relevant(blocked_off_1) as f32),
            ]
        } else {
            [
                (cache[0].relevant_squares.number_of_squares
                    + cache[0].relevant_squares.dist_walked_unhindered) as f32,
                (cache[1].relevant_squares.number_of_squares
                    + cache[1].relevant_squares.dist_walked_unhindered) as f32,
            ]
        };

        let mut score = 0.0;
        let mut finished = false;
        for _ in 0..number_of_averages {
            let scores = self.roll_out_int(small_rng, wall_value, cache, roll_out_new, max_length);
            // we want to map the score non linearly, so that a score of 10 is mapped to 1, and a score of 0 is mapped to 0.5. But around zero the gradient should be a lot steeper,
            // cause those differences are a lot more important.
            //let original_score = (score + 10.0) / 20.0;
            let k = 1.0;
            score += 1.0 / (1.0 + (-k * scores.0).exp());
            finished = scores.1;
        }

        (score / (number_of_averages as f32), finished)
    }

    // Returns the distance in difference between player 0 and player 1
    fn roll_out_int(
        &self,
        small_rng: &mut SmallRng,
        wall_value: fn(u8) -> f32,
        cache: &[NextMovesCache; 2],
        roll_out_new: bool,
        max_length: [f32; 2],
    ) -> (f32, bool) {
        let distance_to_finish_line = [
            cache[0].distances_to_finish.dist[self.pawns[0].position].unwrap() as f32,
            cache[1].distances_to_finish.dist[self.pawns[1].position].unwrap() as f32,
        ];
        let length_diff = if roll_out_new
            // Only want to do this in early to mid game
            && self.pawns[0].number_of_walls_left >= 4
            && self.pawns[1].number_of_walls_left >= 4
        {
            (max_length[1] - max_length[0]) / 6.0
            //0.0
        } else {
            0.0
        };

        let mut walls_penalty = [
            wall_value(self.pawns[1].number_of_walls_left),
            wall_value(self.pawns[0].number_of_walls_left),
        ];

        // In case a pawn has no walls left, the opponents walls increase in value, cause they become way easier to use effectively.
        for i in 0..2 {
            if walls_penalty[i] == 0.0 {
                walls_penalty[(i + 1) % 2] *= 2.0;
            }
        }

        let mut total_distance: Vec<f32> = vec![];
        for i in 0..2 {
            let random_number: f32 = small_rng.gen();
            //let value: f32 =
            //    (walls_penalty[i] * random_number + distance_to_finish_line[i] as f32);
            let value: f32 = walls_penalty[i] * random_number + distance_to_finish_line[i] as f32;

            let value = (value).min(
                (cache[i].relevant_squares.number_of_squares
                    + cache[i].relevant_squares.dist_walked_unhindered) as f32,
            );
            total_distance.push(value);
        }
        // This makes way more sense, cause then it stays consistent at least.
        total_distance[(self.turn + 1) % 2] += 0.5;
        (total_distance[1] - total_distance[0] + length_diff, false)
    }
}

fn ucb_score(wi: f32, ni: u32, big_ni: u32, c: f32, heuristic_score: i8) -> f32 {
    if ni == 0 {
        return f32::INFINITY;
    }
    let ni_f32 = ni as f32; // Convert to f32 for calculation
    let big_ni_f32 = big_ni as f32;

    let heuristic_score = heuristic_score.max(0) as f32;

    (wi as f32 / ni_f32) + c * ((big_ni_f32.ln() / ni_f32).sqrt()) + heuristic_score / ni_f32
}

fn estimate_losses(
    wins: f32,
    total_visits: u32,
    total_parent_visits: u32,
    c: f32,
    threshold_y: f32,
    heuristic_score: i8,
) -> u32 {
    let mut low = 1;
    let mut high = total_parent_visits;

    while low < high {
        let mid = low + (high - low) / 2;
        let new_total_visits = total_visits + mid;
        let new_ucb = ucb_score(
            wins,
            new_total_visits,
            total_parent_visits,
            c,
            heuristic_score,
        );

        if new_ucb < threshold_y {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

// If there are leaves, we select one, run through it and continue.
// If there are no leaves, we select the most promising branch.
fn select_next_node<'a>(
    moves: &'a mut [(Move, MCNode, i8)],
    parent_visits: u32,
    explore_constant: f32,
) -> Option<(
    &'a mut Move,
    &'a mut MCNode,
    u32,
    Option<(f32, Option<usize>)>,
)> {
    // Here we have to hack around because of borrow checker
    select_most_promising_branch(moves, parent_visits, explore_constant)
}

// Return the move with the highest ucb score, and how often we can choose this move before another move could be the most promising.
// So we check after many concurrent loses, this move will be overtaken by the second most promising one.
// For this we assume that the second score won't get much higher, from the number of moves increasing a lot.
// To do this, we check that the parent visits has to be at least 1000. If it is not, we will just return 1.
// Also the number of moves allowed, will be capped at parents_visits / 10.
fn select_most_promising_branch<'a>(
    moves: &'a mut [(Move, MCNode, i8)],
    parent_visits: u32,
    explore_constant: f32,
) -> Option<(
    &'a mut Move,
    &'a mut MCNode,
    u32,
    Option<(f32, Option<usize>)>,
)> {
    let mut max_score = -1000.0;
    let mut highest_scores = (0.0, 0);
    let mut second_score = 0.0;
    let mut chosen_move = None;
    let mut played_out_score: Option<(f32, Option<usize>)> = Some((0.0, None));
    for (index, move_option) in moves.iter_mut().enumerate() {
        let heuristic_score = move_option.2.max(0);
        let score;
        let node_scores;
        let mut leaf = false;
        match &move_option.1 {
            MCNode::Leaf => {
                //if move_option.2 < 0
                //    && (parent_visits <= 10000
                //        || (highest_scores.0 / highest_scores.1 as f32) > 0.5)
                //{
                //    played_out_score = None;
                //    continue;
                //}
                score = 3.0 + heuristic_score as f32;
                node_scores = (0.0, 1);
                played_out_score = None;
                leaf = true;
            }
            MCNode::PlayedOut { scores, .. } => {
                let finish_score = scores.0 / scores.1 as f32;
                // Winning move exists, so this node is played out. Scores below 0.72 we don't consider played out.
                if finish_score > 0.72 {
                    played_out_score = Some((finish_score, Some(index)));
                    chosen_move = Some(move_option);
                    second_score = 0.0;
                    break;
                }
                score = finish_score;
                node_scores = *scores;
                if let Some(played_out_score) = played_out_score.as_mut() {
                    played_out_score.0 = played_out_score.0.max(score);
                }
            }
            MCNode::Branch {
                move_options: _,
                scores,
                ..
            } => {
                played_out_score = None;

                score = ucb_score(
                    scores.0,
                    scores.1,
                    parent_visits,
                    explore_constant,
                    heuristic_score,
                );
                node_scores = *scores;
            }
        }

        if score < max_score {
            second_score = score.max(second_score);
            continue;
        }
        highest_scores = node_scores;
        second_score = second_score.max(max_score);
        max_score = score;
        chosen_move = Some(move_option);
        if leaf {
            break;
        }
    }

    let moves_allowed = if parent_visits >= 1000 {
        std::cmp::min(
            parent_visits / 10,
            estimate_losses(
                highest_scores.0,
                highest_scores.1,
                parent_visits,
                explore_constant,
                second_score,
                0,
            ),
        )
    } else {
        1
    };
    // TODO: change to moves allowed
    chosen_move.map(|c_move| {
        (
            &mut c_move.0,
            &mut c_move.1,
            moves_allowed,
            played_out_score.map(|x| (1.0 - x.0, x.1)),
        )
    })
}

// Here we select the move we're gonna make, which is the one most visited one.
//If there are leaves, it means we either did too little steps, or one of the leaves leads to a win.
pub fn select_robust_best_branch<'a>(
    moves: &Vec<(Move, MCNode, i8)>,
    _board: &Board,
) -> Option<(Move, (f32, u32))> {
    let mut max_visits = 0;
    let mut best_score: f32 = 0.0;
    let mut chosen_move_score = (0.0, 0);
    let mut chosen_move = None;
    let visits_cutoff = moves[(moves.len() - moves.len().min(10))].1.number_visits();
    for move_option in moves.iter() {
        let node_scores;
        match &move_option.1 {
            MCNode::Leaf => {
                continue;
            }
            MCNode::Branch {
                move_options: _,
                scores,
                ..
            } => {
                node_scores = *scores;
            }
            MCNode::PlayedOut { scores, .. } => {
                node_scores = *scores;
            }
        }
        if node_scores.1 >= visits_cutoff {
            println!(
                "{:?}, {:?}, {}, {:.2}",
                move_option.0,
                node_scores,
                move_option.2,
                node_scores.0 / node_scores.1 as f32 * 100.0
            );
        }
        best_score = best_score.max(node_scores.0 / node_scores.1 as f32);
        if node_scores.1 < max_visits {
            continue;
        }
        // NUmber of visits we give to played out nodes
        //if node_scores.1 >= 1_000_000 {
        //    if (node_scores.0 / node_scores.1 as f32) < best_score {
        //        continue;
        //    }
        //}

        chosen_move_score = node_scores;
        max_visits = node_scores.1;
        chosen_move = Some(move_option);
    }
    //println!("{}", best_score);

    chosen_move.map(|c_move| (c_move.0, chosen_move_score))
    //if best_score == (chosen_move_score.0 as f32 / chosen_move_score.1 as f32) {
    //    chosen_move.map(|c_move| (c_move.0, chosen_move_score))
    //} else {
    //    None
    //}
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io::Write;

    use super::*;
    use crate::ai::AIControlledBoard;

    #[derive(Debug)]
    struct MatchResult {
        winner: (u32, Option<f32>, f32, bool, bool, bool, u32),
        winner_started: bool,
        opponent: (u32, Option<f32>, f32, bool, bool, bool, u32),
        // if the game is lasting more then a 100 turns, we will call it a draw
        draw: bool,
    }

    // Here want to add the current score to this string as a new csv line
    fn print_board_score(board: &Board, guessed_score: f32, end_score: f32, to_print: &mut String) {
        let to_append = format!(
            "{}, {:.4}, {:.4} \n",
            board.encode(),
            guessed_score,
            end_score
        );
        to_print.push_str(&to_append);
    }
    fn print_moves_metrics(made_moves: &Vec<(Move, f32)>, file_name: String, end_score: f32) {
        let mut to_print = "".to_string();
        let mut board = Board::new();
        for (_, made_move) in made_moves.iter().enumerate() {
            board.game_move(made_move.0);
            print_board_score(&board, made_move.1, end_score, &mut to_print);
        }
        // Here we want to create the file, if it doesn't exist yet
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_name)
            .unwrap();
        write!(file, "{}", to_print).unwrap();
    }

    fn get_match_result(
        player_a_params: (u32, Option<f32>, f32, bool, bool, bool, u32),
        player_b_params: (u32, Option<f32>, f32, bool, bool, bool, u32),
    ) -> MatchResult {
        let mut board = AIControlledBoard::player_first();
        let player_params = [player_a_params, player_b_params];
        let mut made_moves = vec![];
        loop {
            let players_turn = player_params[board.board.turn % 2];
            let start = Instant::now();
            let wall_value = if let Some(_wall_multiplier) = players_turn.1 {
                // TODO: this is wrong
                |x| x as f32 * 2.0
            } else {
                AIControlledBoard::wall_value
            };

            let pre_calc = PreCalc::new();
            let chosen_move = board
                .relevant_mc_tree
                .decide_move(
                    board.board.clone(),
                    players_turn.0,
                    wall_value,
                    players_turn.2,
                    players_turn.4,
                    players_turn.5,
                    players_turn.6,
                    &pre_calc,
                )
                .unwrap();

            let decision_took = start.elapsed();
            let mut player_0_win_rate: f32 =
                chosen_move.1 .0 as f32 / chosen_move.1 .1 as f32 * 100.0;
            if board.board.turn % 2 == 1 {
                player_0_win_rate = 100.0 - player_0_win_rate;
            }

            made_moves.push((chosen_move.0, player_0_win_rate / 100.0));
            println!(
                "on turn {}, player {:?}, choses move: {:?} in {:?}. Win rate for player 0 is: {:.1}%",
                board.board.turn, players_turn, (chosen_move.0, chosen_move.1),decision_took, player_0_win_rate,
            );
            println!("{}", board.board.encode());
            if MoveResult::Win == board.board.game_move(chosen_move.0) {
                let winner = (board.board.turn + 1) % 2;
                let match_result = MatchResult {
                    winner: player_params[winner],
                    winner_started: winner == 0,
                    opponent: player_params[(board.board.turn) % 2],
                    draw: false,
                };
                let cache = [
                    NextMovesCache::new(&board.board, 0),
                    NextMovesCache::new(&board.board, 1),
                ];
                // the score is the distance of the other pawn to the finish line
                let end_score = board.board.roll_out(
                    &mut SmallRng::from_entropy(),
                    wall_value,
                    &cache,
                    true,
                    1,
                );
                print_moves_metrics(
                    &made_moves,
                    "game_results_board_test_diff.csv".to_string(),
                    end_score.0,
                );
                return match_result;
            }
            if board.board.turn >= 120 {
                let match_result = MatchResult {
                    winner: player_a_params,
                    winner_started: board.board.turn % 2 == 0,
                    opponent: player_b_params,
                    draw: true,
                };
                return match_result;
            }
        }
    }

    #[test]
    fn ai_pick_finish_move() {
        let mut board = AIControlledBoard::player_first();
        board.board.pawns[0].position = Position { row: 6, col: 4 };
        board.board.pawns[1].position = Position { row: 2, col: 8 };
        board.board.pawns[0].goal_row = 8;
        board.board.pawns[1].goal_row = 0;
        board.board.pawns[0].number_of_walls_left = 0;
        board.board.pawns[1].number_of_walls_left = 0;
        board.board.turn = 4;

        let pre_calc = PreCalc::new();
        let chosen_move = board
            .relevant_mc_tree
            .decide_move(
                board.board.clone(),
                10_000,
                AIControlledBoard::wall_value,
                0.3,
                true,
                false,
                100,
                &pre_calc,
            )
            .unwrap();
        println!("chosen move is {:?}", (chosen_move.0, chosen_move.1));
        assert_eq!(chosen_move.0, Move::PawnMove(PawnMove::Up, None));
        board.board.pawns[0].position = Position { row: 7, col: 4 };
        let chosen_move = board
            .relevant_mc_tree
            .decide_move(
                board.board.clone(),
                10_000,
                AIControlledBoard::wall_value,
                0.3,
                true,
                false,
                100,
                &pre_calc,
            )
            .unwrap();
        assert_eq!(chosen_move.0, Move::PawnMove(PawnMove::Up, None));
        println!("chosen move is {:?}", (chosen_move.0, chosen_move.1));
    }

    #[test]
    fn ai_pick_block_move() {
        let mut board = AIControlledBoard::player_first();
        board.board.pawns[0].position = Position { row: 6, col: 4 };
        board.board.pawns[1].position = Position { row: 2, col: 4 };
        board.board.pawns[0].goal_row = 8;
        board.board.pawns[1].goal_row = 0;
        board.board.pawns[0].number_of_walls_left = 1;
        board.board.pawns[1].number_of_walls_left = 0;
        board.board.turn = 4;
        let pre_calc = PreCalc::new();
        let chosen_move = board
            .relevant_mc_tree
            .decide_move(
                board.board.clone(),
                10_000,
                AIControlledBoard::wall_value,
                0.3,
                false,
                false,
                100,
                &pre_calc,
            )
            .unwrap();
        println!(
            "chosen move is {:?}, win_rate is {}",
            (chosen_move.0, chosen_move.1),
            chosen_move.1 .0 / chosen_move.1 .1 as f32
        );
        board.board.pawns[0].position = Position { row: 0, col: 0 };
        board.board.pawns[1].position = Position { row: 1, col: 4 };
        //assert_eq!(chosen_move.0, Move::PawnMove(PawnMove::Up, None));
        let chosen_move = board
            .relevant_mc_tree
            .decide_move(
                board.board.clone(),
                300_000,
                AIControlledBoard::wall_value,
                0.3,
                false,
                false,
                100,
                &pre_calc,
            )
            .unwrap();
        println!(
            "chosen move is {:?}, win_rate is {}",
            (chosen_move.0, chosen_move.1),
            chosen_move.1 .0 / chosen_move.1 .1 as f32
        );
        assert_eq!(chosen_move.0, Move::PawnMove(PawnMove::Up, None));
    }

    fn test_new_logic() {
        let mut player_a_params = (60_000, None, 0.5, true, false, true, 100);
        let player_a_clone = player_a_params;
        let mut player_b_params = (60_000, None, 0.5, true, false, false, 1);

        let win_sum = std::sync::Arc::new(std::sync::Mutex::new(0.0));
        let draw_sum = std::sync::Arc::new(std::sync::Mutex::new(0.0));
        let total_sum = std::sync::Arc::new(std::sync::Mutex::new(0));

        // we want atomic f32s
        let number_of_threads = 1;
        let number_of_games = 1;
        let mut threads = vec![];
        for _ in 0..number_of_threads {
            let win_sum = win_sum.clone();
            let draw_sum = draw_sum.clone();
            let total_sum = total_sum.clone();
            let handle = std::thread::spawn(move || {
                for _ in 1..(number_of_games / number_of_threads + 1) {
                    std::mem::swap(&mut player_a_params, &mut player_b_params);
                    let match_result = get_match_result(player_a_params, player_b_params);
                    println!("{:#?}", match_result);
                    println!(
                        "{}; {}; {}",
                        match_result.winner.0, match_result.opponent.0, match_result.winner_started
                    );

                    let mut win_sum = win_sum.lock().unwrap();
                    let mut draw_sum = draw_sum.lock().unwrap();
                    let mut total_sum = total_sum.lock().unwrap();
                    *total_sum += 1;
                    if match_result.draw {
                        *draw_sum += 1.0;
                    } else {
                        if match_result.winner == player_a_clone {
                            *win_sum += 1.0;
                        }
                    }
                    println!(
                        "After {} games: Win rate for player 0 is: {:.1}%, draw rate is {:.1}%",
                        *total_sum,
                        *win_sum / *total_sum as f32 * 100.0,
                        *draw_sum / *total_sum as f32 * 100.0,
                    );
                }
            });
            threads.push(handle);
        }
        for handle in threads {
            handle.join().unwrap();
        }
        // Now we want to serialize the cache using bincode to the file called moves_cache
    }

    fn test_ai_battle() {
        let opponents = vec![20_000, 40_000, 60_000, 120_000, 200_000, 400_000];
        let mut match_ups_seen = HashSet::new();
        for opponent in &opponents {
            match_ups_seen.insert((*opponent, *opponent));
        }
        let mut match_results = vec![];

        let number_of_matches = 10;
        println!("winner; loser ; winner_started");
        'outer: for player_a in &opponents {
            for player_b in &opponents {
                if match_ups_seen.get(&(*player_a, *player_b)).is_some() {
                    continue 'outer;
                }
                match_ups_seen.insert((*player_b, *player_a));
                for i in 0..number_of_matches {
                    // we want to switch up who plays first
                    let number_steps = if i % 2 == 0 {
                        [player_a, player_b]
                    } else {
                        [player_b, player_a]
                    };
                    let match_result = get_match_result(
                        (*number_steps[0], Some(2.0), 0.3, false, false, true, 1),
                        (*number_steps[1], None, 0.3, false, false, true, 1),
                    );
                    println!(
                        "{}; {}; {}",
                        match_result.winner.0, match_result.opponent.0, match_result.winner_started
                    );
                    match_results.push(match_result);
                }
            }
        }
    }

    #[test]
    fn test_wrong_calc() {
        let mut board = AIControlledBoard::player_first();
        let walls = [
            (WallDirection::Horizontal, Position { row: 2, col: 2 }),
            (WallDirection::Horizontal, Position { row: 5, col: 3 }),
            (WallDirection::Horizontal, Position { row: 5, col: 5 }),
        ];
        for (dir, loc) in walls {
            board.board.place_wall(dir, loc);
        }
        board.board.pawns[0].position = Position { row: 3, col: 4 };
        board.board.pawns[0].number_of_walls_left = 9;
        board.board.pawns[1].position = Position { row: 5, col: 4 };
        board.board.pawns[1].number_of_walls_left = 8;
        board.board.turn = 8;

        let cache = [
            NextMovesCache::new(&board.board, 0),
            NextMovesCache::new(&board.board, 1),
        ];
        let mut sum = 0.0;
        let iterations = 100;
        for _ in 0..iterations {
            let roll_out_winner = board.board.roll_out(
                &mut SmallRng::from_entropy(),
                AIControlledBoard::wall_value,
                &cache,
                true,
                100,
            );
            println!("roll out value: {:?}", roll_out_winner);
            sum += roll_out_winner.0;
        }
        println!("roll out sum {}", sum / iterations as f32);
        let roll_out_winner = board.board.roll_out(
            &mut SmallRng::from_entropy(),
            AIControlledBoard::wall_value,
            &cache,
            false,
            100,
        );
        println!("{:?}", roll_out_winner);
    }

    #[test]
    fn black_to_win_puzzle() {
        // See quoridor discord
        let board =
            Board::decode("32;3G5;2A5;A1h;C1v;D2h;F2h;H2h;C3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;A7h;C7h")
                .unwrap();
        let mut ai_controlled = AIControlledBoard::from_board(board);
        let chosen_move = ai_controlled.ai_move(300_000, &PreCalc::new());
        assert_eq!(chosen_move.0, Move::PawnMove(PawnMove::Left, None));

        let board =
            Board::decode("34;3F5;2A4;A1h;C1v;D2h;F2h;H2h;C3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;A7h;C7h")
                .unwrap();

        let mut ai_controlled = AIControlledBoard::from_board(board);
        let chosen_move = ai_controlled.ai_move(400_000, &PreCalc::new());

        assert_eq!(
            chosen_move.0,
            Move::Wall(WallDirection::Horizontal, Position { row: 5, col: 7 })
        );
    }
    #[test]
    fn black_to_win_puzzle_2() {
        // See quoridor discord
        let board = Board::decode(
            "19;1G7;1G3;C2v;G2h;A3h;C3h;E3v;D4v;E4h;C5h;E5v;G5h;H5v;A6h;C6h;E6h;H6h;C7v;F7h;H7h",
        )
        .unwrap();
        let mut ai_controlled = AIControlledBoard::from_board(board);
        let chosen_move = ai_controlled.ai_move(300_000, &PreCalc::new());

        assert_eq!(
            chosen_move.0,
            Move::Wall(WallDirection::Horizontal, Position { row: 1, col: 3 })
        );
    }

    #[test]
    fn white_wins_puzzle() {
        // See quoridor discord
        let board = Board::decode(
            "17;1F4;3D4;F1h;H1h;A2h;C2h;E2h;F3v;D4h;G4h;B5h;D5h;E5v;F5h;H5h;D6v;F6v;D8v",
        )
        .unwrap();
        let mut ai_controlled = AIControlledBoard::from_board(board);
        let chosen_move = ai_controlled.ai_move(300_000, &PreCalc::new());

        assert_eq!(
            chosen_move.0,
            Move::Wall(WallDirection::Horizontal, Position { row: 2, col: 3 })
        );
    }

    #[test]
    fn player_2_wins_puzzle() {
        // See quoridor discord
        let board = Board::decode(
            "17;1I7;3H5;B1v;C2h;E2h;G2h;B3v;D3h;F3v;E4v;H4v;B5v;C5h;E5h;G5h;D6v;E6h;D8v",
        )
        .unwrap();
        let mut ai_controlled = AIControlledBoard::from_board(board);
        let chosen_move = ai_controlled.ai_move(300_000, &PreCalc::new());

        // Can also be wall vertical, multiple options basically
        //assert_eq!(chosen_move.0, Move::PawnMove(PawnMove::Down, None));
    }
    #[test]
    fn hard_puzzle() {
        // See quoridor discord
        let board = Board::decode("17;6D4;6D6;C2h;C4v;E5h;G5h;D6h;F6h;H6h;C7v").unwrap();
        let mut ai_controlled = AIControlledBoard::from_board(board);
        let chosen_move = ai_controlled.ai_move(10_300_000, &PreCalc::new());

        // To decide on this move, it needs a lottttttt of calculations
        assert_eq!(
            chosen_move.0,
            Move::Wall(WallDirection::Horizontal, Position { row: 4, col: 2 })
        );
    }

    #[test]
    fn easier_puzzle() {
        // See quoridor discord
        let board =
            Board::decode("12;5E8;4E3;A3v;B3v;C3h;E3v;A4h;D4v;E5v;D6v;F6h;H6h;E8h").unwrap();
        let mut ai_controlled = AIControlledBoard::from_board(board);
        let chosen_move = ai_controlled.ai_move(3200_000, &PreCalc::new());

        // To decide on this move, it needs a lottttttt of calculations
        assert!(
            chosen_move.0 == Move::Wall(WallDirection::Horizontal, Position { row: 7, col: 1 })
                || chosen_move.0
                    == Move::Wall(WallDirection::Vertical, Position { row: 7, col: 2 })
        );
    }
    #[test]
    fn test_only_bad_pawn_moves_but_still_win() {
        let board = Board::decode(
            "26;1E5;2F5;G2v;A3h;C3h;E3h;G3h;D4v;G4h;C5v;D5h;G5h;H5v;D6h;F6v;G6v;H6h;E7v;F8v",
        )
        .unwrap();

        let mut ai_controlled = AIControlledBoard::from_board(board);
        let chosen_move = ai_controlled.ai_move(60_000, &PreCalc::new());

        assert!(chosen_move.0 == Move::PawnMove(PawnMove::Right, Some(PawnMove::Right)));

        let scores_opponent = ai_controlled.relevant_mc_tree.mc_node.scores();
        assert!((scores_opponent.0 / scores_opponent.1 as f32) < 0.5);
    }
}
