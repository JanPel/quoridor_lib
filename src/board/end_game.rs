use super::*;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum JumpResult {
    ZeroJumpsOne,
    OneJumpsZero,
}

impl Board {
    pub fn roll_out_finish(&self, cache: &[NextMovesCache; 2]) -> Option<f32> {
        let previous_pawn = self.pawns[(self.turn + 1) % 2];

        let distance_to_finish_line = [
            cache[0].distances_to_finish.dist[self.pawns[0].position].unwrap() as f32,
            cache[1].distances_to_finish.dist[self.pawns[1].position].unwrap() as f32,
        ];

        if previous_pawn.goal_row == previous_pawn.position.row {
            if self.turn % 2 == 0 {
                return Some(-distance_to_finish_line[0]);
            } else {
                return Some(distance_to_finish_line[1]);
            }
        }
        if distance_to_finish_line[self.turn % 2] == 1.0 {
            if self.turn % 2 == 0 {
                return Some(distance_to_finish_line[1]);
            } else {
                return Some(-distance_to_finish_line[0]);
            }
        }
        if (self.pawns[1].number_of_walls_left == 0
            || cache[0].relevant_squares.number_of_squares <= 1)
            && (self.pawns[0].number_of_walls_left == 0
                || cache[1].relevant_squares.number_of_squares <= 1)
        {
            if let Some(res) = self.winner_when_no_walls_or_single_paths(cache) {
                return Some(res);
            }
        }
        if let Some(res) = self.one_wall_roll_out(cache) {
            return Some(res);
        }
        None
    }

    // How long the distance to the finish from this pawn is wit the best placed wall.
    fn max_dist_finish(
        &self,
        pawn: Pawn,
        allowed_walls: &AllowedWalls,
        distances_to_finish: &DistancesToFinish,
    ) -> i8 {
        let mut res = distances_to_finish.dist[pawn.position].unwrap();
        for row in 0..2 {
            for col in 0..2 {
                let row = row as i8 + pawn.position.row - 1;
                let col = col as i8 + pawn.position.col - 1;
                if row < 0 || row > 7 || col < 0 || col > 7 {
                    continue;
                }

                let pos = Position { row, col };
                for dir in [WallDirection::Horizontal, WallDirection::Vertical] {
                    {
                        if allowed_walls[(dir, pos)] == WallType::Impossible {
                            continue;
                        }
                        if distances_to_finish.wall_parrallel(dir, pos) {
                            continue;
                        }

                        if distances_to_finish.pawn_short_side_wall(dir, pos, pawn.position) {
                            continue;
                        }
                        // pos is on closer side to finish of wall
                        let mut open_routes = self.open_routes.clone();
                        open_routes.update_open(dir, pos);
                        if let Some(dist) = open_routes.find_path_for_pawn_to_dest_row(pawn, true) {
                            res = res.max(dist as i8);
                        }
                    }
                }
            }
        }
        res
    }

    fn dijkstra_distance_to_finish_1_wall_opponent_no_jumps(
        &self,
        pawn: Pawn,
        allowed_walls: &AllowedWalls,
        distances_to_finish: &DistancesToFinish,
        pawns_turn: bool,
    ) -> (i8, bool) {
        let mut queue = BinaryHeap::new();
        let mut distances: [[Vec<(i8, i8)>; 9]; 9] = Default::default();
        let mut max_dist_cache = [[None; 9]; 9];
        let moves_order = if pawn.goal_row == 8 {
            PAWN_MOVES_DOWN_LAST
        } else {
            PAWN_MOVES_UP_LAST
        };
        let goal_row = pawn.goal_row;
        let mut best_option_length_for_walker = 100;
        let mut wall_used_best_option = false;

        queue.push((
            -distances_to_finish.dist[pawn.position].unwrap(),
            pawn.position,
        ));
        distances[pawn.position] = vec![(
            0,
            self.max_dist_finish(pawn, &allowed_walls, &distances_to_finish),
        )];
        if pawns_turn {
            // The pawn can already do its first step, which the other pawn can never block.
            for pawn_move in moves_order {
                if !self.open_routes.is_open(pawn.position, pawn_move) {
                    continue;
                }
                let next = pawn.position.add_move(pawn_move);
                if next.row == pawn.goal_row {
                    return (1, false);
                }

                queue.push((-distances_to_finish.dist[next].unwrap(), next));
                let pawn = Pawn {
                    position: next,
                    goal_row,
                    number_of_walls_left: 0,
                };
                distances[next] = vec![(
                    1,
                    self.max_dist_finish(pawn, &allowed_walls, &distances_to_finish) + 1,
                )];
            }
        }

        'outer: while !queue.is_empty() {
            let (_, current) = queue.pop().unwrap();
            for pawn_move in moves_order {
                if !self.open_routes.is_open(current, pawn_move) {
                    continue;
                }
                let next = current.add_move(pawn_move);
                if next.row == goal_row {
                    for (walked_distance, total_distance) in &distances[current] {
                        best_option_length_for_walker =
                            best_option_length_for_walker.min(*total_distance);
                        if best_option_length_for_walker == *total_distance {
                            wall_used_best_option = *total_distance != *walked_distance + 1;
                        }
                    }
                    continue 'outer;
                }

                if best_option_length_for_walker < 100 {
                    distances[current].retain(|x| x.1 < best_option_length_for_walker);
                    if distances[current].is_empty() {
                        continue;
                    }
                }

                let dist_from_next = &mut max_dist_cache[next];
                if dist_from_next.is_none() {
                    *dist_from_next = Some(self.max_dist_finish(
                        Pawn {
                            position: next,
                            goal_row,
                            number_of_walls_left: 0,
                        },
                        allowed_walls,
                        distances_to_finish,
                    ));
                }
                let mut next_distances = vec![];
                std::mem::swap(&mut distances[next], &mut next_distances);
                // TODO: look into swapping
                let mut added = 0;
                for (dist_walked, total_distance) in &distances[current] {
                    let new_dist_walked = *dist_walked + 1;
                    let new_total_distance =
                        (new_dist_walked + dist_from_next.unwrap()).max(*total_distance);
                    if new_total_distance >= best_option_length_for_walker {
                        continue;
                    }
                    if next_distances
                        .iter()
                        .any(|x| (x.0 <= new_dist_walked && x.1 <= new_total_distance))
                    {
                        continue;
                    }
                    next_distances
                        .retain(|x| !(x.0 >= new_dist_walked && x.1 >= new_total_distance));
                    added += 1;
                    next_distances.push((new_dist_walked, new_total_distance));
                }
                std::mem::swap(&mut distances[next], &mut next_distances);
                if added >= 1 {
                    queue.push((-distances_to_finish.dist[next].unwrap(), next));
                }
            }
        }
        (best_option_length_for_walker, wall_used_best_option)
    }

    // This logic could still go wrong, if a wall is also blocking our own route_to finish
    fn is_overlap(
        &self,
        pawn_with_walls: Pawn,
        pawn_without_walls: Pawn,
        dist_finish_pawn_with_walls: DistancesToFinish,
        dist_finish_pawn_without_walls: DistancesToFinish,
        rel_squares_pawn_no_walls: &RelevantSquares,
    ) -> bool {
        let routes_to_finish =
            self.routes_to_finish(pawn_with_walls.position, dist_finish_pawn_with_walls, 0);

        let forced_walk_lenght = rel_squares_pawn_no_walls.dist_walked_unhindered;
        for row in 0..9 {
            for col in 0..9 {
                let pos = Position { row, col };
                if routes_to_finish[pos] == -3 {
                    continue;
                }
                if rel_squares_pawn_no_walls.squares[pos].is_none() {
                    continue;
                }

                let distance_from_pawn_without_walls;
                if rel_squares_pawn_no_walls.squares[pos].unwrap() != -1 {
                    distance_from_pawn_without_walls =
                        rel_squares_pawn_no_walls.squares[pos].unwrap() + forced_walk_lenght;
                } else {
                    distance_from_pawn_without_walls =
                        dist_finish_pawn_without_walls.dist[pawn_without_walls.position].unwrap()
                            - dist_finish_pawn_without_walls.dist[pos].unwrap();
                }
                if distance_from_pawn_without_walls > routes_to_finish[pos] + 2 {
                    continue;
                }
                return true;
            }
        }
        false
    }

    // Here we check if only one pawn has a one wall left. And if there paths don't overlap
    pub fn one_wall_roll_out(&self, cache: &[NextMovesCache; 2]) -> Option<f32> {
        let (pawn_without_walls, pawn_no_walls_cache, index) =
            if self.pawns[1].number_of_walls_left == 1 && self.pawns[0].number_of_walls_left == 0 {
                if self.is_overlap(
                    self.pawns[1],
                    self.pawns[0],
                    cache[1].distances_to_finish,
                    cache[0].distances_to_finish,
                    &cache[0].relevant_squares,
                ) {
                    return None;
                }
                (self.pawns[0], &cache[0], 0)
            } else if self.pawns[0].number_of_walls_left == 1
                && self.pawns[1].number_of_walls_left == 0
            {
                if self.is_overlap(
                    self.pawns[0],
                    self.pawns[1],
                    cache[0].distances_to_finish,
                    cache[1].distances_to_finish,
                    &cache[1].relevant_squares,
                ) {
                    return None;
                }
                (self.pawns[1], &cache[1], 1)
            } else {
                return None;
            };
        let (dist, wall_used) = self.dijkstra_distance_to_finish_1_wall_opponent_no_jumps(
            pawn_without_walls,
            &pawn_no_walls_cache.allowed_walls_for_pawn,
            &pawn_no_walls_cache.distances_to_finish,
            self.pawns[self.turn % 2].number_of_walls_left == 0,
        );
        let turn_effect = if (self.turn % 2) == 0 { 0.5 } else { -0.5 };
        let score = if index == 0 {
            let dist_1 = cache[1].distances_to_finish.dist[self.pawns[1].position].unwrap()
                + wall_used as i8;
            dist_1 - dist
        } else {
            let dist_0 = cache[0].distances_to_finish.dist[self.pawns[0].position].unwrap()
                + wall_used as i8;
            dist - dist_0
        };
        let score = score as f32 + turn_effect;

        return Some(score.signum() * score.abs().ceil());
    }

    fn winner_when_no_walls(
        &self,
        cache: &[NextMovesCache; 2],
        turn_index: i8,
        turn_take_effect: (i8, i8),
    ) -> Option<f32> {
        let distance_to_finish_line = [
            cache[0].distances_to_finish.dist[self.pawns[0].position].unwrap() as f32,
            cache[1].distances_to_finish.dist[self.pawns[1].position].unwrap() as f32,
        ];

        if let Some(jump_effect) =
            self.overlapping_routes_to_finish(cache, turn_index, turn_take_effect)
        {
            let score = if turn_index == 0 {
                // Over nadenken.. TODOOOOO!!!!!
                distance_to_finish_line[1] + turn_take_effect.1 as f32 + 0.5
                    - (distance_to_finish_line[0] + turn_take_effect.0 as f32)
                    + jump_effect as f32
            } else {
                (distance_to_finish_line[1] + turn_take_effect.1 as f32)
                    - (distance_to_finish_line[0] + 0.5 + turn_take_effect.0 as f32)
                    + jump_effect as f32
            };
            return Some(score.signum() * score.abs().ceil());
        }
        None
    }

    pub fn winner_when_no_walls_or_single_paths(&self, cache: &[NextMovesCache; 2]) -> Option<f32> {
        let _distance_to_finish_line = [
            cache[0].distances_to_finish.dist[self.pawns[0].position].unwrap() as f32,
            cache[1].distances_to_finish.dist[self.pawns[1].position].unwrap() as f32,
        ];

        let walls_diff =
            self.pawns[0].number_of_walls_left as i8 - self.pawns[1].number_of_walls_left as i8;

        let turn_index = (self.turn % 2) as i8;
        if walls_diff == 0 {
            self.winner_when_no_walls(cache, turn_index, (0, 0))
        } else {
            // TOOO Many Variables......
            let score_no_wall_used = self.winner_when_no_walls(cache, turn_index, (0, 0))?;

            // If we give away the turn, no extra logic needed, but getting the turn, costs us 1 walks.
            let turn_take_cost = match (turn_index, walls_diff.signum()) {
                (0, 1) => (0, 0),
                (0, -1) => (-2, 0),
                (1, 1) => (0, -2),
                (1, -1) => (0, 0),
                (_, _) => panic!("shouldn't happen"),
            };

            let score_with_turn_switched =
                self.winner_when_no_walls(cache, (turn_index + 1) % 2, turn_take_cost)?;

            // This logic is wrong, cause you can't get the turn forward, only backward with using a wall.
            if walls_diff >= 0 {
                Some(score_no_wall_used.max(score_with_turn_switched))
            } else {
                Some(score_no_wall_used.min(score_with_turn_switched))
            }
        }
    }

    fn overlapping_routes_to_finish(
        &self,
        cache: &[NextMovesCache; 2],
        turn_index: i8,
        turn_jump_effects: (i8, i8),
    ) -> Option<i8> {
        let distance_to_finish_line = [
            cache[0].distances_to_finish.dist[self.pawns[0].position].unwrap(),
            cache[1].distances_to_finish.dist[self.pawns[1].position].unwrap(),
        ];

        let opponent_distances_to_our_finish = [
            cache[0].distances_to_finish.dist[self.pawns[1].position].unwrap_or(100),
            cache[1].distances_to_finish.dist[self.pawns[0].position].unwrap_or(100),
        ];
        // In this case the pawns will never see each other
        if distance_to_finish_line[0] < opponent_distances_to_our_finish[0]
            || distance_to_finish_line[1] < opponent_distances_to_our_finish[1]
        {
            return Some(0);
        }
        let routes_to_finish_0 = self.routes_to_finish(
            self.pawns[0].position,
            cache[0].distances_to_finish,
            turn_jump_effects.0,
        );
        let routes_to_finish_1 = self.routes_to_finish(
            self.pawns[1].position,
            cache[1].distances_to_finish,
            turn_jump_effects.1,
        );

        let mut overlap_positions = vec![];
        for row in 0..9 {
            for col in 0..9 {
                let pos = Position { row, col };
                if routes_to_finish_0[pos] == -3 || routes_to_finish_1[pos] == -3 {
                    continue;
                }
                let distance_0 = routes_to_finish_0[pos];
                let distance_1 = routes_to_finish_1[pos];
                if distance_0 == distance_1 {
                    if distance_0 <= 0 || distance_1 <= 0 {
                        return None;
                    }
                    let jump_result = if turn_index == 0 {
                        JumpResult::OneJumpsZero
                    } else {
                        JumpResult::ZeroJumpsOne
                    };
                    overlap_positions.push((pos, distance_0, distance_1, jump_result));
                }
                if distance_0 - 1 == distance_1 && turn_index == 0 {
                    if distance_0 <= 0 || distance_1 <= 0 {
                        return None;
                    }
                    overlap_positions.push((pos, distance_0, distance_1, JumpResult::ZeroJumpsOne));
                }
                if distance_0 == distance_1 - 1 && turn_index == 1 {
                    if distance_0 <= 0 || distance_1 <= 0 {
                        return None;
                    }
                    overlap_positions.push((pos, distance_0, distance_1, JumpResult::OneJumpsZero));
                }
            }
        }
        if overlap_positions.is_empty() {
            return Some(0);
        }
        let res = self.clear_jump_effects(
            [routes_to_finish_0, routes_to_finish_1],
            cache,
            overlap_positions,
        );
        res.map(|x| x.2)
    }

    fn jump_effects(
        &self,
        routes_to_finish: [[[i8; 9]; 9]; 2],
        cache: &[NextMovesCache; 2],
        overlap_positions: Vec<(Position, i8, i8, JumpResult)>,
    ) -> Vec<((Position, Position), JumpResult, i8)> {
        // First we calculate from which positions the jumping pawn can jump

        let mut effects = vec![];
        for overlap_pos in overlap_positions {
            let mut pawn_positions = vec![];
            for pawn_move in PAWN_MOVES_DOWN_LAST {
                if !self.open_routes.is_open(overlap_pos.0, pawn_move) {
                    continue;
                }
                let jump_pos = overlap_pos.0.add_move(pawn_move);
                match overlap_pos.3 {
                    JumpResult::ZeroJumpsOne => {
                        if routes_to_finish[0][jump_pos] == routes_to_finish[0][overlap_pos.0] - 1 {
                            pawn_positions.push((jump_pos, overlap_pos.0));
                        }
                    }
                    JumpResult::OneJumpsZero => {
                        if routes_to_finish[1][jump_pos] == routes_to_finish[1][overlap_pos.0] - 1 {
                            pawn_positions.push((overlap_pos.0, jump_pos));
                        }
                    }
                }
            }
            for (pos_pawn_0, pos_pawn_1) in pawn_positions {
                let mut board = self.clone();
                board.pawns[0].position = pos_pawn_0;
                board.pawns[1].position = pos_pawn_1;
                let (old_pos, next_positions) = match overlap_pos.3 {
                    JumpResult::ZeroJumpsOne => {
                        board.turn = 0;
                        (pos_pawn_0, board.next_pawn_moves())
                    }
                    JumpResult::OneJumpsZero => {
                        board.turn = 1;
                        (pos_pawn_1, board.next_pawn_moves())
                    }
                };
                // Only add the best jump Cause now the pawn that moving can choose, and greedy is the best way, cause after this it's just walking to the finish
                let mut best_jump: Option<((Position, Position), JumpResult, i8, i8)> = None;
                for (_, next_pos) in next_positions {
                    let (jump_effect, cmp_value) = match overlap_pos.3 {
                        JumpResult::ZeroJumpsOne => {
                            // How much closer did pawn zero get to the finish?
                            let cmp_value = cache[0].distances_to_finish.dist[old_pos].unwrap()
                                - cache[0].distances_to_finish.dist[next_pos].unwrap();
                            (cmp_value - 1, cmp_value)
                        }
                        JumpResult::OneJumpsZero => {
                            // How much closer did pawn one get to the finish? And then negative
                            let cmp_value = cache[1].distances_to_finish.dist[old_pos].unwrap()
                                - cache[1].distances_to_finish.dist[next_pos].unwrap();

                            (-(cmp_value - 1), cmp_value)
                        }
                    };
                    if best_jump.is_none() || cmp_value > best_jump.unwrap().3 {
                        best_jump = Some((
                            (pos_pawn_0, pos_pawn_1),
                            overlap_pos.3,
                            jump_effect,
                            cmp_value,
                        ));
                    }
                }
                let best_jump = best_jump.unwrap();
                effects.push((best_jump.0, best_jump.1, best_jump.2));

                // Now we check what would be the best pawn_position for the jumping pawn.
            }
        }
        effects
    }

    fn clear_jump_effects(
        &self,
        routes_to_finish: [[[i8; 9]; 9]; 2],
        cache: &[NextMovesCache; 2],
        overlap_positions: Vec<(Position, i8, i8, JumpResult)>,
    ) -> Option<((Position, Position), JumpResult, i8)> {
        let effects = self.jump_effects(routes_to_finish, cache, overlap_positions);
        if effects.len() != 1 {
            // Not so clear, need to look into it deeper
            return None;
        }

        let ((pawn_zero, pawn_one), jump_result, jump_effect) = effects[0];

        let (distance, distance_from_pawn, to_escape_pos) = if jump_effect == 1 {
            // Good jumps for pawn zero, so we want to check if pawn one can escape
            let distance_walked = routes_to_finish[1][pawn_one];
            // check if routes to finish has another position with the same distance from start.
            (distance_walked, &routes_to_finish[1], pawn_one)
        } else if jump_effect == -1 {
            let distance_walked = routes_to_finish[0][pawn_zero];
            // check if routes to finish has another position with the same distance from start.
            (distance_walked, &routes_to_finish[0], pawn_zero)
        } else {
            // THere is a jump but it has no effect on distances
            return Some(((pawn_zero, pawn_one), jump_result, jump_effect));
        };
        // Can the pawn thats not benefitting escape a bad jump?
        for row in 0..9 {
            for col in 0..9 {
                let pos = Position { row, col };
                if pos == to_escape_pos {
                    continue;
                }
                if distance_from_pawn[pos] == distance {
                    // Could also return that a jump is not possible or something?
                    return None;
                }
            }
        }

        return Some(((pawn_zero, pawn_one), jump_result, jump_effect));
    }
}

#[cfg(test)]
mod test {
    use std::sync::atomic::AtomicU32;

    #[cfg(not(target_arch = "wasm32"))]
    use std::time::Instant;
    #[cfg(target_arch = "wasm32")]
    use web_time::Instant;

    use rand::SeedableRng;
    use std::sync::Arc;

    use super::*;
    #[test]
    fn test_finish_dist_2_paths_1_wall() {
        let mut board = Board::decode("7;6E1;7E9;F4v;H4v;F6v;H6v;A7h;C7h;E7h").unwrap();
        board.pawns[0].position = Position { row: 2, col: 8 };
        let cache = NextMovesCache::new(&board, 0);
        let start = Instant::now();
        let max_dist = board.dijkstra_distance_to_finish_1_wall_opponent_no_jumps(
            board.pawns[0],
            &cache.allowed_walls_for_pawn,
            &cache.distances_to_finish,
            false,
        );
        println!("{:?} in {:?}", max_dist, start.elapsed());
        assert_eq!(max_dist, (16, true));
    }
    #[test]
    fn test_finish_dist_2_paths_only_one_blockable_wall() {
        let mut board = Board::decode("9;5E1;6E9;G4v;H4v;A6h;G6v;H6v;A7v;B7h;D7h;F7h").unwrap();
        board.pawns[0].position = Position { row: 3, col: 7 };
        let cache = NextMovesCache::new(&board, 0);
        let start = Instant::now();
        let max_dist = board.dijkstra_distance_to_finish_1_wall_opponent_no_jumps(
            board.pawns[0],
            &cache.allowed_walls_for_pawn,
            &cache.distances_to_finish,
            false,
        );
        println!("{:?} in {:?}", max_dist, start.elapsed());
        assert_eq!(max_dist, (10, true));
    }
    #[test]
    fn test_finish_dist_2_paths_only_one_blockable_wall_2() {
        let mut board =
            Board::decode("11;4E1;5E9;G3v;H3v;G5v;H5v;A6h;A7v;B7h;D7h;F7h;G7v;H7v").unwrap();
        board.pawns[0].position = Position { row: 5, col: 7 };
        let cache = NextMovesCache::new(&board, 0);
        let start = Instant::now();
        let max_dist = board.dijkstra_distance_to_finish_1_wall_opponent_no_jumps(
            board.pawns[0],
            &cache.allowed_walls_for_pawn,
            &cache.distances_to_finish,
            false,
        );
        println!("{:?} in {:?}", max_dist, start.elapsed());
        assert_eq!(max_dist, (12, false));
    }

    #[test]
    fn test_finish_dist_3_wide_classic() {
        let mut board = Board::decode("6;0G6;1E9;F4v;F6v;A7h;C7h;E7h;H7h").unwrap();
        board.pawns[0].position = Position { row: 5, col: 7 };
        let cache = NextMovesCache::new(&board, 0);
        let start = Instant::now();
        let max_dist = board.dijkstra_distance_to_finish_1_wall_opponent_no_jumps(
            board.pawns[0],
            &cache.allowed_walls_for_pawn,
            &cache.distances_to_finish,
            true,
        );
        println!("{:?} in {:?}", max_dist, start.elapsed());
        assert_eq!(max_dist, (8, true));
    }
    #[test]
    fn test_finish_roll_out() {
        // This board has overlap, so we can not conclude anything
        let mut board =
            Board::decode("14;3E1;3E9;D1v;D3v;E4h;F5v;H5v;B6h;D6h;A7h;C7v;E7h;F7v;H7v;B8h;D8h")
                .unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].number_of_walls_left = 1;
        let start = Instant::now();

        let max_dist = board.one_wall_roll_out(&cache);

        println!("{:?} in {:?}", max_dist, start.elapsed());
        assert!(max_dist.is_none());

        let mut board = Board::decode("9;5E1;6E9;D1v;D3v;E4h;F4v;H4v;F6v;H6v;F8v;H8v").unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].number_of_walls_left = 1;

        board.turn = 0;
        let start = Instant::now();

        let max_dist = board.one_wall_roll_out(&cache);

        println!("{:?} in {:?}", max_dist, start.elapsed());
        assert_eq!(max_dist, Some(-12.0));

        board.turn = 1;
        let start = Instant::now();

        let max_dist = board.one_wall_roll_out(&cache);

        println!("{:?} in {:?}", max_dist, start.elapsed());
        assert_eq!(max_dist, Some(-13.0));
    }

    #[test]
    fn test_winner_no_walls() {
        let mut board = Board::new();
        board.pawns[0].position = Position { row: 0, col: 6 };
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].position = Position { row: 8, col: 4 };
        board.pawns[1].number_of_walls_left = 0;
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        let start = Instant::now();
        println!(
            "{}",
            board
                .winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0))
                .unwrap()
        );
        println!("{:?}", start.elapsed());
        assert_eq!(
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0)),
            Some(1.0)
        );

        let mut board = Board::new();
        board.pawns[0].position = Position { row: 0, col: 6 };
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].position = Position { row: 7, col: 4 };
        board.pawns[1].number_of_walls_left = 0;

        assert_eq!(
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0)),
            Some(-1.0)
        );
        println!(
            "{}",
            board
                .winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0))
                .unwrap()
        );

        let mut board = Board::new();
        board.pawns[0].position = Position { row: 0, col: 4 };
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].position = Position { row: 7, col: 4 };
        board.pawns[1].number_of_walls_left = 0;

        let start = Instant::now();
        println!(
            "{:?}",
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0))
        );
        println!("{:?}", start.elapsed());
        assert_eq!(
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0)),
            Some(1.0)
        );
    }

    #[test]
    fn test_winner_walls_left() {
        let board = Board::decode("10;0E1;0E9;D1v;E1v;D3v;E3v;D5v;E5v;D7v;E7v").unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        assert_eq!(
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0)),
            Some(-1.0)
        );

        let board = Board::decode("10;0E1;0E8;D1v;E1v;D3v;E3v;D5v;E5v;D7v;E7v").unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        println!("THE WRONG CALC!!!!!!!!!!!!!!!!!!!!!!-----------------------------");
        assert_eq!(
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0)),
            Some(1.0)
        );

        let board = Board::decode("10;0E1;2E8;D1v;E1v;D3v;E3v;D5v;E5v;D7v;E7v").unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        assert_eq!(
            board.winner_when_no_walls_or_single_paths(&cache),
            Some(-1.0)
        );

        let board =
            Board::decode("0;1E1;0E9;D1v;E1v;D3v;E3v;D5v;E5v;D7v;E7v;B8v;C8h;F8h;G8v").unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        assert_eq!(
            board.winner_when_no_walls_or_single_paths(&cache),
            Some(1.0)
        );
        assert_eq!(
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0)),
            Some(-1.0)
        );
    }

    #[test]
    fn test_not_able_to_determine_winner() {
        let mut board = Board::decode("5;7E1;8E9;E1h;D2h;E2v;D3v;E3h").unwrap();
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].number_of_walls_left = 0;
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        assert_eq!(
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0)),
            None
        );

        //
        let mut board = Board::decode("6;6E1;7E9;D1v;E1h;D2h;E2v;D3v;E3h;F4v").unwrap();
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].number_of_walls_left = 0;
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        assert_eq!(
            board.winner_when_no_walls(&cache, (board.turn % 2) as i8, (0, 0)),
            Some(-1.0)
        );
    }
    #[test]
    fn test_able_to_determine_winner() {
        let mut board = Board::decode("66;1B2;0A6;B1h;A2v;B3v;D3h;F3h;H3h;A4v;D4v;B5v;D5h;F5v;G5h;D6v;E6h;G6v;H6h;B7v;C8h;D8v").unwrap();
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].number_of_walls_left = 0;
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        assert_eq!(
            board
                .roll_out(
                    &mut SmallRng::from_entropy(),
                    |x| 2.0 * x as f32,
                    &cache,
                    false,
                    100,
                )
                .1,
            true
        );
    }

    use crate::ai::*;
    use crate::move_stats::PreCalc;
    #[test]
    fn test_speedup() {
        let board = Board::decode("66;1B2;0A6;B1h;A2v;B3v;D3h;F3h;H3h;A4v;D4v;B5v;D5h;F5v;G5h;D6v;E6h;G6v;H6h;B7v;C8h;D8v").unwrap();
        let mut mc_node = MCNode::Leaf;
        let mut small_rng = SmallRng::from_entropy();
        let mut timings = Timings::default();
        let mut calc_cache = CalcCache::zero();
        let last_visit_count = Arc::new(AtomicU32::new(0));

        let start = Instant::now();
        recursive_monte_carlo(
            board.clone(),
            &mut mc_node,
            &mut small_rng,
            600,
            1,
            AIControlledBoard::wall_value,
            0.5,
            true,
            &mut timings,
            None,
            true,
            100,
            &mut calc_cache,
            &last_visit_count,
            &PreCalc::new(),
        );
        println!("monte carlo took: {:?}", start.elapsed());
        println!("{:?}", timings);
    }

    #[test]
    fn test_speedup_1_wall() {
        let mut board = Board::decode("1;0E1;1E9;D1v;D3v;E4h;F4v;H4v;F6v;H6v;F8v;H8v").unwrap();
        board.pawns[0].number_of_walls_left = 0;
        board.pawns[1].number_of_walls_left = 1;

        let mut mc_node = MCNode::Leaf;
        let mut small_rng = SmallRng::from_entropy();
        let mut timings = Timings::default();
        let mut calc_cache = CalcCache::zero();
        let last_visit_count = Arc::new(AtomicU32::new(0));

        let start = Instant::now();
        recursive_monte_carlo(
            board.clone(),
            &mut mc_node,
            &mut small_rng,
            80_000,
            1,
            AIControlledBoard::wall_value,
            0.5,
            true,
            &mut timings,
            None,
            true,
            100,
            &mut calc_cache,
            &last_visit_count,
            &PreCalc::new(),
        );
        println!("monte carlo took: {:?}", start.elapsed());
        match mc_node {
            MCNode::Branch { scores, .. } => {
                println!("scores: {:?}", scores);
                //println!("move options: {:?}", move_options);
            }
            MCNode::Leaf => {
                println!("LEAF");
            }
            MCNode::PlayedOut { scores, .. } => {
                println!("PLAYED OUT: scores: {:?}", scores);
            }
        };
        println!("{:?}", timings);
    }

    #[test]
    fn test_end_game_puzzle() {
        let board = Board::decode(
            "41;0H5;2G6;B2v;D2v;E2v;C3h;F3h;H3h;D4v;F4h;D5h;E5v;A6h;C6h;D6v;G6h;D7h;H7v;E8v;H8h",
        )
        .unwrap();

        let mut mc_node = MCNode::Leaf;
        let mut small_rng = SmallRng::from_entropy();
        let mut timings = Timings::default();
        let mut calc_cache = CalcCache::zero();
        let last_visit_count = Arc::new(AtomicU32::new(0));

        let start = Instant::now();
        recursive_monte_carlo(
            board.clone(),
            &mut mc_node,
            &mut small_rng,
            80_000,
            0,
            AIControlledBoard::wall_value,
            0.5,
            true,
            &mut timings,
            None,
            true,
            100,
            &mut calc_cache,
            &last_visit_count,
            &PreCalc::new(),
        );
        println!("monte carlo took: {:?}", start.elapsed());

        let move_chosen =
            select_robust_best_branch(mc_node.move_options().unwrap(), &board).unwrap();
        assert_eq!(move_chosen.0, Move::PawnMove(PawnMove::Down, None));
        assert!(move_chosen.1 .0 / move_chosen.1 .1 as f32 > 0.5);

        let board = Board::decode(
            "43;0F5;2G5;B2v;D2v;E2v;C3h;F3h;H3h;D4v;F4h;D5h;E5v;A6h;C6h;D6v;G6h;D7h;H7v;E8v;H8h",
        )
        .unwrap();

        let mut mc_node = MCNode::Leaf;
        let mut small_rng = SmallRng::from_entropy();
        let mut timings = Timings::default();
        let mut calc_cache = CalcCache::zero();

        let start = Instant::now();
        recursive_monte_carlo(
            board.clone(),
            &mut mc_node,
            &mut small_rng,
            80_000,
            0,
            AIControlledBoard::wall_value,
            0.5,
            true,
            &mut timings,
            None,
            true,
            100,
            &mut calc_cache,
            &last_visit_count,
            &PreCalc::new(),
        );
        println!("monte carlo took: {:?}", start.elapsed());

        let move_chosen =
            select_robust_best_branch(mc_node.move_options().unwrap(), &board).unwrap();
        assert_eq!(
            move_chosen.0,
            Move::Wall(WallDirection::Horizontal, Position { row: 4, col: 5 })
        );
        assert!(move_chosen.1 .0 / move_chosen.1 .1 as f32 > 0.5);

        let board = Board::decode(
            "45;0H5;1G5;B2v;D2v;E2v;C3h;F3h;H3h;D4v;F4h;D5h;E5v;F5h;A6h;C6h;D6v;G6h;D7h;H7v;E8v;H8h",
        )
        .unwrap();

        let mut mc_node = MCNode::Leaf;
        let mut small_rng = SmallRng::from_entropy();
        let mut timings = Timings::default();
        let mut calc_cache = CalcCache::zero();

        let start = Instant::now();
        recursive_monte_carlo(
            board.clone(),
            &mut mc_node,
            &mut small_rng,
            80_000,
            0,
            AIControlledBoard::wall_value,
            0.5,
            true,
            &mut timings,
            None,
            true,
            100,
            &mut calc_cache,
            &last_visit_count,
            &PreCalc::new(),
        );
        println!("monte carlo took: {:?}", start.elapsed());

        let move_chosen =
            select_robust_best_branch(mc_node.move_options().unwrap(), &board).unwrap();
        assert_eq!(
            move_chosen.0,
            Move::Wall(WallDirection::Horizontal, Position { row: 6, col: 5 })
        );
        assert!(move_chosen.1 .0 / move_chosen.1 .1 as f32 > 0.5);
    }

    #[test]
    fn test_winner_straight_line() {
        let board =
            Board::decode("18;4E3;3E7;D1v;E1v;D3v;E3v;D5v;E5v;D7v;E7v;B8v;C8h;F8h;G8v").unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        assert_eq!(
            board.winner_when_no_walls_or_single_paths(&cache),
            Some(1.0)
        );

        // This is the same scenario as above, just a few more steps ahead.
        // In this case the winner should be pawn number 1, cause they have the turn and will take that to jump.
        let board =
            Board::decode("19;4E5;3E6;D1v;E1v;D3v;E3v;D5v;E5v;D7v;E7v;B8v;C8h;F8h;G8v").unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        assert_eq!(
            board.winner_when_no_walls_or_single_paths(&cache),
            //Some(-1.0), is real answer, but None is fine as well.
            None
        );
    }

    #[test]
    fn test_winner_both_jumps_bad() {
        let board = Board::decode(
            "19;0I5;0F9;H1v;E2h;D3v;F3v;H3v;D4h;G4h;E5v;H5h;B6h;F6v;G6v;E7v;G7h;C8v;D8h;F8v",
        )
        .unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        assert_eq!(
            board.winner_when_no_walls_or_single_paths(&cache),
            //Some(-1.0), is real answer, but None is fine as well.
            Some(5.0)
        );

        let board = Board::decode(
            "16;0I3;1F9;H1v;E2h;D3v;F3v;H3v;D4h;G4h;E5v;H5h;F6v;G6v;E7v;G7h;C8v;D8h;F8v",
        )
        .unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        assert_eq!(
            board.winner_when_no_walls_or_single_paths(&cache),
            Some(4.0)
        );
    }

    #[test]
    fn test_end_game_wrong() {
        let board = Board::decode(
            "36;2F5;1A4;A1h;C1v;D2h;F2h;H2h;A3h;C3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;F6h;A7h;C7h",
        )
        .unwrap();

        let mut mc_node = MCNode::Leaf;
        let mut small_rng = SmallRng::from_entropy();
        let mut timings = Timings::default();
        let mut calc_cache = CalcCache::zero();
        let last_visit_count = Arc::new(AtomicU32::new(0));

        let start = Instant::now();
        recursive_monte_carlo(
            board.clone(),
            &mut mc_node,
            &mut small_rng,
            100,
            0,
            AIControlledBoard::wall_value,
            0.5,
            true,
            &mut timings,
            None,
            true,
            100,
            &mut calc_cache,
            &last_visit_count,
            &PreCalc::new(),
        );

        let move_chosen =
            select_robust_best_branch(mc_node.move_options().unwrap(), &board).unwrap();
        if let Some(move_options) = mc_node.move_options() {
            for mc_node in move_options {
                match &mc_node.1 {
                    MCNode::Branch { scores, .. } => {
                        println!("move: {:?}, scores: {:?}", mc_node.0, scores);
                    }
                    MCNode::Leaf => {
                        println!("move: {:?}, LEAF", mc_node.0);
                    }
                    MCNode::PlayedOut { scores, .. } => {
                        println!("move: {:?}, played out: {:?}", mc_node.0, scores);
                    }
                };
            }
        }
        println!("For board: {}", board.encode());
        println!(
            "{:?}, win chance: {}",
            move_chosen,
            move_chosen.1 .0 / move_chosen.1 .1 as f32
        );
        println!("monte carlo took: {:?}", start.elapsed());

        let board = Board::decode(
            "37;1F5;1A4;A1h;C1v;B2h;D2h;F2h;H2h;A3h;C3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;F6h;A7h;C7h",
        )
        .unwrap();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        println!(
            "Finish roll out result: {:?}",
            board.roll_out_finish(&cache)
        );
        let mut mc_node = MCNode::Leaf;
        let mut calc_cache = CalcCache::zero();
        let last_visit_count = Arc::new(AtomicU32::new(0));

        recursive_monte_carlo(
            board.clone(),
            &mut mc_node,
            &mut small_rng,
            86,
            0,
            AIControlledBoard::wall_value,
            0.5,
            true,
            &mut timings,
            None,
            true,
            100,
            &mut calc_cache,
            &last_visit_count,
            &PreCalc::new(),
        );
        if let Some(move_options) = mc_node.move_options() {
            for mc_node in move_options {
                println!("{:?}: {:?}", mc_node.0, mc_node.1);
            }
        }

        println!("Finished board: {}", board.encode());
        println!("{:?}", mc_node);

        let move_chosen =
            select_robust_best_branch(mc_node.move_options().unwrap(), &board).unwrap();

        assert_eq!(move_chosen.0, Move::PawnMove(PawnMove::Right, None));
        assert!(move_chosen.1 .0 / move_chosen.1 .1 as f32 > 0.5);
    }

    #[test]
    fn test_winner_end_game_wall_placement_too_late() {
        let board = Board::decode(
            "47;0H8;2C3;D3h;F3h;G3v;H3v;A4h;C4v;E4v;B5h;D5v;F5v;G5v;A6h;C6h;E6h;H6h;E7v;F7h;E8h",
        )
        .unwrap();

        let mut mc_node = MCNode::Leaf;
        let mut small_rng = SmallRng::from_entropy();
        let mut timings = Timings::default();
        let mut calc_cache = CalcCache::zero();
        let last_visit_count = Arc::new(AtomicU32::new(0));

        recursive_monte_carlo(
            board.clone(),
            &mut mc_node,
            &mut small_rng,
            86,
            0,
            AIControlledBoard::wall_value,
            0.5,
            true,
            &mut timings,
            None,
            true,
            100,
            &mut calc_cache,
            &last_visit_count,
            &PreCalc::new(),
        );
        if let Some(move_options) = mc_node.move_options() {
            for mc_node in move_options {
                println!("{:?}: {:?}", mc_node.0, mc_node.1);
            }
        }

        println!("Finished board: {}", board.encode());
        println!("{:?}", mc_node);

        let move_chosen =
            select_robust_best_branch(mc_node.move_options().unwrap(), &board).unwrap();

        assert!((move_chosen.1 .0 / move_chosen.1 .1 as f32) < 0.5);
    }
}
