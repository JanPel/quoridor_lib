mod end_game;
mod walls;

use std::collections::{BinaryHeap, VecDeque};

use rand::prelude::SliceRandom;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use crate::ai::PlayerMoveResult;

use arraydeque::ArrayDeque;

pub use walls::*;

pub enum PocketCheckResponse {
    Pocket([[Option<i8>; 9]; 9]),
    // If the bool is true, we reached the other side
    NoPocket(bool),
}

// TODO: make  exclude function into trait for performance, then we don't need to do line check
const MAX_LINE_LENGTH: usize = 4;

pub const PAWN_MOVES_DOWN_LAST: [PawnMove; 4] = [
    PawnMove::Up,
    PawnMove::Left,
    PawnMove::Right,
    PawnMove::Down,
];

const PAWN_MOVES_UP_LAST: [PawnMove; 4] = [
    PawnMove::Down,
    PawnMove::Left,
    PawnMove::Right,
    PawnMove::Up,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PawnMove {
    Up,
    Down,
    Left,
    Right,
}

impl PawnMove {
    pub fn orthogonal_moves(&self) -> Vec<PawnMove> {
        match self {
            PawnMove::Up | PawnMove::Down => vec![PawnMove::Left, PawnMove::Right],
            PawnMove::Left | PawnMove::Right => vec![PawnMove::Up, PawnMove::Down],
        }
    }

    pub fn orthogonal_walls(&self) -> WallDirection {
        match self {
            PawnMove::Up | PawnMove::Down => WallDirection::Horizontal,
            PawnMove::Left | PawnMove::Right => WallDirection::Vertical,
        }
    }

    pub fn opposite_move(&self) -> PawnMove {
        match self {
            PawnMove::Up => PawnMove::Down,
            PawnMove::Down => PawnMove::Up,
            PawnMove::Left => PawnMove::Right,
            PawnMove::Right => PawnMove::Left,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Position {
    pub row: i8,
    pub col: i8,
}

impl std::ops::Index<(WallDirection, Position)> for Walls {
    type Output = bool;

    fn index(&self, index: (WallDirection, Position)) -> &Self::Output {
        let (wall_direction, pos) = index;
        match wall_direction {
            WallDirection::Horizontal => &self.horizontal[pos],
            WallDirection::Vertical => &self.vertical[pos],
        }
    }
}

impl std::ops::IndexMut<(WallDirection, Position)> for Walls {
    fn index_mut(&mut self, index: (WallDirection, Position)) -> &mut Self::Output {
        let (wall_direction, pos) = index;
        match wall_direction {
            WallDirection::Horizontal => &mut self.horizontal[pos],
            WallDirection::Vertical => &mut self.vertical[pos],
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Index<Position> for [[T; COLS]; ROWS] {
    type Output = T;

    fn index(&self, index: Position) -> &Self::Output {
        let Position { row, col } = index;
        &self[row as usize][col as usize]
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::IndexMut<Position> for [[T; COLS]; ROWS] {
    fn index_mut(&mut self, index: Position) -> &mut Self::Output {
        let Position { row, col } = index;
        &mut self[row as usize][col as usize]
    }
}

pub struct CheckPocketResponseNew {
    wall_allowed: bool,
    is_pocket: bool,
    // If we reached the otherside, the number indicates in how many steps we reached the other side of the wall
    reached_otherside: Option<i8>,
    pawn_seen: bool,
    goal_row_seen: bool,
    // The area that is on this part of the wall
    area_seen: [[Option<u8>; 9]; 9],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WallEffect {
    DistanceOtherside(i8),
    AreaLeftPawn([[Option<u8>; 9]; 9]),
}
impl WallEffect {
    fn wall_score(&self, rel_squares: RelevantSquares) -> i8 {
        match self {
            WallEffect::DistanceOtherside(steps) => *steps,
            WallEffect::AreaLeftPawn(area) => {
                let mut number_of_squares_left_relevant = 0;
                for row in 0..9 {
                    for col in 0..9 {
                        let pos = Position { row, col };
                        if area[pos].is_some()
                            && rel_squares.squares[pos].is_some()
                            && rel_squares.squares[pos] != Some(-1)
                        {
                            number_of_squares_left_relevant += 1;
                        }
                    }
                }
                rel_squares.number_of_squares - number_of_squares_left_relevant
            }
        }
    }
}

impl Position {
    pub fn encode(&self) -> String {
        let columns = "ABCDEFGHI";
        let rows = "123456789";
        format!(
            "{}{}",
            columns.chars().nth(self.col as usize).unwrap(),
            rows.chars().nth(self.row as usize).unwrap()
        )
    }
    pub fn decode(input: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let columns = "ABCDEFGHI";
        let rows = "123456789";
        let col = columns
            .find(input.chars().nth(0).unwrap())
            .ok_or("Invalid column")?;
        let row = rows
            .find(input.chars().nth(1).unwrap())
            .ok_or("Invalid row")?;
        Ok(Self {
            row: row as i8,
            col: col as i8,
        })
    }
    pub fn add_move(&self, pawn_move: PawnMove) -> Position {
        let (row, col) = match pawn_move {
            PawnMove::Up => (self.row + 1, self.col),
            PawnMove::Down => (self.row - 1, self.col),
            PawnMove::Left => (self.row, self.col - 1),
            PawnMove::Right => (self.row, self.col + 1),
        };
        Position { row, col }
    }

    pub fn substract_pos(&self, other: &Self) -> Option<PawnMove> {
        match (other.row - self.row, other.col - self.col) {
            (1, 0) => Some(PawnMove::Up),
            (0, 1) => Some(PawnMove::Right),
            (-1, 0) => Some(PawnMove::Down),
            (0, -1) => Some(PawnMove::Left),
            _ => None,
        }
    }

    fn add_pawn_moves(&self, pawn_move: PawnMove, second_move: Option<PawnMove>) -> Position {
        let mut position = self.add_move(pawn_move);
        if let Some(pawn_move) = second_move {
            position = position.add_move(pawn_move);
        }
        position
    }

    // For most positions two walls can block the shortest path, but for some positions only one wall can block this move.
    // This is because walls have length 2; If only wall could block this move, we just return that wall twice.
    fn relevant_walls(&self, pawn_move: PawnMove) -> [(WallDirection, Position); 2] {
        let Position { row, col } = *self;
        let wall_col_1;
        let wall_col_2;
        let wall_row_1;
        let wall_row_2;
        match pawn_move {
            PawnMove::Up | PawnMove::Down => {
                if col == 0 {
                    wall_col_1 = col;
                    wall_col_2 = col;
                } else if col == 8 {
                    wall_col_1 = col - 1;
                    wall_col_2 = col - 1;
                } else {
                    wall_col_1 = col - 1;
                    wall_col_2 = col;
                };
                if pawn_move == PawnMove::Up {
                    wall_row_1 = row;
                    wall_row_2 = row;
                } else {
                    wall_row_1 = row - 1;
                    wall_row_2 = row - 1;
                }
            }
            PawnMove::Right | PawnMove::Left => {
                if row == 0 {
                    wall_row_1 = row;
                    wall_row_2 = row;
                } else if row == 8 {
                    wall_row_1 = row - 1;
                    wall_row_2 = row - 1;
                } else {
                    wall_row_1 = row - 1;
                    wall_row_2 = row;
                };
                if pawn_move == PawnMove::Right {
                    wall_col_1 = col;
                    wall_col_2 = col;
                } else {
                    wall_col_1 = col - 1;
                    wall_col_2 = col - 1;
                }
            }
        };
        let wall_direction = match pawn_move {
            PawnMove::Up | PawnMove::Down => WallDirection::Horizontal,
            PawnMove::Right | PawnMove::Left => WallDirection::Vertical,
        };
        [
            (
                wall_direction,
                Position {
                    row: wall_row_1,
                    col: wall_col_1,
                },
            ),
            (
                wall_direction,
                Position {
                    row: wall_row_2,
                    col: wall_col_2,
                },
            ),
        ]
    }

    // This is not ex
    fn move_to(&self, next_position: Position) -> Option<PawnMove> {
        if next_position.row == self.row + 1 && next_position.col == self.col {
            Some(PawnMove::Up)
        } else if next_position.row == self.row - 1 && next_position.col == self.col {
            Some(PawnMove::Down)
        } else if next_position.col == self.col + 1 && next_position.row == self.row {
            Some(PawnMove::Right)
        } else if next_position.col == self.col - 1 && next_position.row == self.row {
            Some(PawnMove::Left)
        } else {
            None
        }
    }

    fn point_mirror_point(&self) -> Position {
        Position {
            row: 8 - self.row,
            col: 8 - self.col,
        }
    }
}

impl From<(usize, usize)> for Position {
    fn from((row, col): (usize, usize)) -> Self {
        Self {
            row: row as i8,
            col: col as i8,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pawn {
    pub position: Position,
    pub goal_row: i8,
    pub number_of_walls_left: u8,
}

impl Pawn {
    fn new_bottom() -> Self {
        Self {
            position: Position { row: 0, col: 4 },
            goal_row: 8,
            number_of_walls_left: 10,
        }
    }

    fn new_top() -> Self {
        Self {
            position: Position { row: 8, col: 4 },
            goal_row: 0,
            number_of_walls_left: 10,
        }
    }

    fn encode(&self) -> String {
        format!("{}{}", self.number_of_walls_left, self.position.encode())
    }
    fn decode(input: &str, pawn_index: i8) -> Result<Self, Box<dyn std::error::Error>> {
        let pos_start = input.len() - 2;
        let number_of_walls_left = input[0..pos_start].parse::<u8>()?;
        let position = Position::decode(&input[pos_start..])?;
        let goal_row = if pawn_index == 0 { 8 } else { 0 };
        Ok(Self {
            position,
            goal_row,
            number_of_walls_left,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelevantSquares {
    pub squares: [[Option<i8>; 9]; 9],
    pub number_of_squares: i8,
    pub distance_zero_square: Position,
    pub dist_walked_unhindered: i8,
}

impl RelevantSquares {
    pub const fn zero() -> Self {
        Self {
            squares: [[None; 9]; 9],
            number_of_squares: 72,
            distance_zero_square: Position { row: 0, col: 0 },
            dist_walked_unhindered: 0,
        }
    }
    pub fn one() -> Self {
        Self {
            squares: [[Some(1); 9]; 9],
            number_of_squares: 72,
            distance_zero_square: Position { row: 0, col: 0 },
            dist_walked_unhindered: 0,
        }
    }

    pub fn forced_walk_length(&self) -> i8 {
        let mut forced_walk_length = 0;
        for row in 0..9 {
            for col in 0..9 {
                let pos = Position { row, col };
                if self.squares[pos].is_some() && self.squares[pos] == Some(-1) {
                    forced_walk_length += 1;
                }
            }
        }
        forced_walk_length
    }
    pub fn number_of_left_relevant(&self, pockets: [[i8; 9]; 9]) -> i8 {
        let mut count = 0;
        for row in 0..9 {
            for col in 0..9 {
                let pos = Position { row, col };
                if (self.squares[pos].is_some() && pockets[pos] <= 1)
                    || self.squares[pos] == Some(-1)
                {
                    count += 1;
                }
            }
        }
        if count == 0 {
            self.number_of_squares
        } else {
            count
        }
    }
    // We want to print all the squares with rows on seperate rows, with a '|' as seperator between distances and for None we want to print '.'
    pub fn pretty_print(&self) {
        let mut buffer = String::new();
        for row in 0..9 {
            for col in 0..9 {
                buffer.push_str("|");
                if let Some(distance) = self.squares[row][col] {
                    // we want the distance to take up three digits
                    buffer.push_str(&format!("{:>3}", distance));
                } else {
                    buffer.push_str(" . ");
                }
            }
            buffer.push_str("|\n");
        }
        println!("{}", buffer);
    }

    fn walk_to_finish(walked_steps: Vec<Position>) -> Self {
        let distance_zero_square = *walked_steps.last().unwrap();
        let mut relevant_squares = RelevantSquares {
            number_of_squares: 0,
            squares: [[None; 9]; 9],
            distance_zero_square,
            dist_walked_unhindered: walked_steps.len() as i8 - 1,
        };
        for position in &walked_steps[0..(walked_steps.len() - 1)] {
            // TODO: ERROR If some of these ones is Some, we need to recalculate the distances...
            if relevant_squares.squares[*position].is_none() {
                // -1 means that its a squares where we can walk unhindered to
                relevant_squares.squares[*position] = Some(-1);
            }
        }
        return relevant_squares;
    }
    fn new(
        board: &Board,
        pawn_index: usize,
        distances_to_finish: &DistancesToFinish,
        allowed_walls_for_pawn: &AllowedWalls,
    ) -> Self {
        let mut rel_squares = Self::one();
        let mut pawn = board.pawns[pawn_index];

        // If the other pawn has no walls left, no squares are relevant, cause we will just walk shortest path.
        if board.pawns[(pawn_index + 1) % 2].number_of_walls_left == 0 {
            return RelevantSquares {
                number_of_squares: 0,
                squares: [[None; 9]; 9],
                distance_zero_square: pawn.position,
                dist_walked_unhindered: distances_to_finish.dist[pawn.position].unwrap(),
            };
        }
        if let Some((walked_steps, excluded)) = board.open_routes.pawn_walk_unhindered(
            pawn,
            allowed_walls_for_pawn,
            &distances_to_finish,
        ) {
            pawn.position = *walked_steps.last().unwrap();

            // We can walk directly to the end, so we don't need to do anything else.
            if walked_steps.last().unwrap().row == pawn.goal_row {
                return Self::walk_to_finish(walked_steps);
            }

            rel_squares.exclude(excluded);
            for position in &walked_steps[0..(walked_steps.len() - 1)] {
                rel_squares.squares[*position] = Some(-1);
            }
            rel_squares.dist_walked_unhindered = walked_steps.len() as i8 - 1;
        }
        rel_squares = rel_squares.update_distances(
            pawn.position,
            &board.open_routes,
            pawn.goal_row,
            allowed_walls_for_pawn,
            true,
        );
        rel_squares.exclude_squares(&board.open_routes, &distances_to_finish.dist, pawn);

        rel_squares
    }
    // In this function we want to use the dijkstra algorithm to determine all the distances from the pawn. For all squares that are not None.
    fn update_distances(
        self,
        new_position: Position,
        open_routes: &OpenRoutes,
        goal_row: i8,
        allowed_walls: &AllowedWalls,
        keep_minus_ones: bool,
    ) -> Self {
        if self.squares[new_position] == Some(-1) {
            return self;
        }
        let mut new = [[None; 9]; 9];
        new[new_position] = Some(0);
        let mut queue = VecDeque::new();
        let mut count = 1;
        let moves_order = if goal_row == 8 {
            PAWN_MOVES_DOWN_LAST
        } else {
            PAWN_MOVES_UP_LAST
        };
        queue.push_back(new_position);
        'outer: while let Some(position) = queue.pop_front() {
            let distance = new[position];
            let distance = distance.unwrap();
            'inner: for pawn_move in moves_order {
                if !open_routes.is_open(position, pawn_move) {
                    continue 'inner;
                }
                let new_position = position.add_move(pawn_move);
                if new_position.row == goal_row {
                    // If this path path to the goal line can't be blocked, The other pawn_moves are not relevant. Cause pawn can never be forced to go there.
                    for relevant_wall in position.relevant_walls(pawn_move) {
                        if allowed_walls[relevant_wall].is_allowed() {
                            continue 'inner;
                        }
                    }
                    // THis skip makes distances incorrect..
                    // But we update the distances latery anyway, so doesn't matter.
                    continue 'outer;
                }
                if self.squares[new_position].is_none() || self.squares[new_position] == Some(-1) {
                    continue 'inner;
                }
                if let Some(old_distance) = new[new_position] {
                    if old_distance <= distance + 1 {
                        continue 'inner;
                    }
                }

                count += 1;
                new[new_position] = Some(distance + 1);
                queue.push_back(new_position);
            }
        }
        // Here we update all the forced walk squares
        if keep_minus_ones {
            for row in 0..9 {
                for col in 0..9 {
                    let pos = Position { row, col };
                    if self.squares[pos] == Some(-1) {
                        new[pos] = Some(-1);
                    }
                }
            }
        }
        Self {
            squares: new,
            number_of_squares: count,
            distance_zero_square: new_position,
            dist_walked_unhindered: self.dist_walked_unhindered,
        }
    }

    fn exclude(&mut self, excluded: [[Option<i8>; 9]; 9]) {
        for i in 0..9 {
            for j in 0..9 {
                let pos = Position { row: i, col: j };
                if !excluded[pos].is_none()
                    && self.squares[pos].is_some()
                    && self.squares[pos] != Some(-1)
                {
                    self.squares[pos] = None;
                    self.number_of_squares -= 1;
                }
            }
        }
    }

    pub fn exclude_from_position(
        &mut self,
        position: Position,
        pawn: Pawn,
        distances_to_finish: &[[Option<i8>; 9]; 9],
        open_routes: &OpenRoutes,
    ) -> Option<[[Option<i8>; 9]; 9]> {
        for pawn_move in PAWN_MOVES_DOWN_LAST {
            if !open_routes.is_open(position, pawn_move) {
                continue;
            }
            let next = position.add_move(pawn_move);

            if next == pawn.position {
                continue;
            }
            if self.squares[next].is_none() || self.squares[next] == Some(-1) {
                continue;
            }
            if distances_to_finish[next].unwrap() <= distances_to_finish[position].unwrap() {
                continue;
            }

            match open_routes.check_if_line_only_entrance(
                [position; MAX_LINE_LENGTH],
                pawn,
                pawn_move,
                &self,
                &distances_to_finish,
            ) {
                PocketCheckResponse::Pocket(pocket) => {
                    for row in 0..9 {
                        for col in 0..9 {
                            let pos = Position { row, col };
                            if pocket[pos].is_some() {
                                if self.squares[pos].is_some() && self.squares[pos] != Some(-1) {
                                    self.squares[pos] = None;
                                    self.number_of_squares -= 1;
                                }
                            }
                        }
                    }
                }
                PocketCheckResponse::NoPocket(false) => continue,
                PocketCheckResponse::NoPocket(true) => continue,
                // TODO: performance Think about conclusions here..
                //PocketCheckResponse::NoPocket(true) => return None,
            };
        }
        None
    }

    pub fn exclude_squares(
        &mut self,
        open_routes: &OpenRoutes,
        distances_from_finish: &[[Option<i8>; 9]; 9],
        pawn: Pawn,
    ) {
        // we will loop through the distances from pawn, starting from far away to close by.
        for row in 0..9 {
            for col in 0..9 {
                let position = Position { row, col };
                let distance = self.squares[position];
                if distance.is_none() {
                    continue;
                }
                let distance = distance.unwrap();
                if distance == -1 {
                    continue;
                }
                self.exclude_from_position(position, pawn, distances_from_finish, open_routes);
            }
        }
    }

    fn new_cache(
        mut self,
        game_move: Move,
        new_board: &Board,
        old_pawn: Pawn,
        pawn_index: usize,
        did_cache_pawn_move: bool,
        distances_to_finish: &DistancesToFinish,
        allowed_walls: &AllowedWalls,
    ) -> Self {
        let new_pawn = new_board.pawns[pawn_index];
        if new_board.pawns[(pawn_index + 1) % 2].number_of_walls_left == 0 {
            return Self {
                squares: [[None; 9]; 9],
                number_of_squares: 0,
                distance_zero_square: new_pawn.position,
                dist_walked_unhindered: distances_to_finish.dist[new_pawn.position].unwrap(),
            };
        }
        match game_move {
            Move::PawnMove(first_step, second_step) => {
                if !did_cache_pawn_move {
                    return self;
                }

                let new_position = new_pawn.position;
                // The pawn was stepped outside of the relevant squares (a dumb move), but this might mean that a lot more squares are relevant again.
                if self.squares[new_position].is_none() {
                    return Self::new(new_board, pawn_index, distances_to_finish, allowed_walls);
                } else if self.squares[new_position] == Some(-1)
                    || self.squares[new_position] == Some(0)
                {
                    if let Some(second_step) = second_step {
                        let skipped_position = old_pawn.position.add_move(first_step);
                        self.dist_walked_unhindered -= 1;
                        let old_value = self.squares[skipped_position].take();
                        if old_value == None {
                            // we skipped outside of relevant squares
                            // So we must have gone diagonally.
                            let diagonal_skipped = old_pawn.position.add_move(second_step);
                            self.squares[diagonal_skipped] = None;
                        }
                    }

                    self.dist_walked_unhindered -= 1;
                    self.squares[old_pawn.position] = None;
                    return self;
                }

                if new_position.row == old_pawn.goal_row {
                    // If the pawn is on the goal row, we can remove all squares that are not on the goal row.
                    self.squares = [[None; 9]; 9];
                    self.number_of_squares = 0;
                    return self;
                }

                // Now we want to check whether the pawn moved to a square where the pawn can walk unhindered.
                if let Some((walked_to_unhindered, excluded)) = new_board
                    .open_routes
                    .pawn_walk_unhindered(new_pawn, allowed_walls, &distances_to_finish)
                {
                    if walked_to_unhindered.last().unwrap().row == new_pawn.goal_row {
                        return Self::walk_to_finish(walked_to_unhindered);
                    }
                    self.exclude(excluded);
                    self = self.update_distances(
                        *walked_to_unhindered.last().unwrap(),
                        &new_board.open_routes,
                        old_pawn.goal_row,
                        allowed_walls,
                        false,
                    );
                    for &pos in &walked_to_unhindered[0..(walked_to_unhindered.len() - 1)] {
                        self.squares[pos] = Some(-1);
                    }
                    self.dist_walked_unhindered = walked_to_unhindered.len() as i8 - 1;
                    self
                    // Put the walked unhindered squares to -1 and zero
                } else {
                    self = self.update_distances(
                        new_pawn.position,
                        &new_board.open_routes,
                        old_pawn.goal_row,
                        allowed_walls,
                        false,
                    );
                    self.dist_walked_unhindered = 0;
                    // Now we want to see if the old position of the pawn is still relevant, or whether they can be excluded.

                    self.exclude_from_position(
                        new_pawn.position,
                        new_pawn,
                        &distances_to_finish.dist,
                        &new_board.open_routes,
                    );

                    if let Some(_) = second_step {
                        let old_position = old_pawn.position;
                        let next_position = old_position.add_move(first_step);
                        if self.squares[next_position].is_some()
                            && self.squares[next_position] != Some(-1)
                        {
                            self.exclude_from_position(
                                next_position,
                                new_pawn,
                                &distances_to_finish.dist,
                                &new_board.open_routes,
                            );
                        }
                    }
                    self
                }
            }
            Move::Wall(dir, pos) => {
                let mut temp_pawn = old_pawn.clone();
                temp_pawn.position = self.distance_zero_square;
                let walked_to_unhindered = new_board.open_routes.pawn_walk_unhindered(
                    temp_pawn,
                    allowed_walls,
                    &distances_to_finish,
                );
                let mut updated_distances = false;
                if let Some((walked_to_unhindered, excluded)) = walked_to_unhindered {
                    if walked_to_unhindered.last().unwrap().row == new_pawn.goal_row {
                        if self.dist_walked_unhindered == 0 {
                            return Self::walk_to_finish(walked_to_unhindered);
                        } else {
                            let mut to_return = Self::walk_to_finish(walked_to_unhindered);
                            // We need to add the old walk
                            for row in 0..9 {
                                for col in 0..9 {
                                    let pos = Position { row, col };
                                    if self.squares[pos] == Some(-1) {
                                        to_return.squares[pos] = Some(-1);
                                    }
                                }
                            }
                            to_return.dist_walked_unhindered += self.dist_walked_unhindered;
                            return to_return;
                        }
                    }
                    self.exclude(excluded);
                    self = self.update_distances(
                        *walked_to_unhindered.last().unwrap(),
                        &new_board.open_routes,
                        old_pawn.goal_row,
                        allowed_walls,
                        true,
                    );
                    for &pos in &walked_to_unhindered[0..(walked_to_unhindered.len() - 1)] {
                        self.squares[pos] = Some(-1);
                    }
                    self.dist_walked_unhindered += walked_to_unhindered.len() as i8 - 1;
                    updated_distances = true;
                }
                if closest_square_distance(pos, &self.squares).is_none() {
                    return self;
                }
                if !updated_distances {
                    self = self.update_distances(
                        self.distance_zero_square,
                        &new_board.open_routes,
                        old_pawn.goal_row,
                        allowed_walls,
                        true,
                    );
                }
                if new_board
                    .walls
                    .connect_on_two_points(dir, pos, Some(old_pawn.goal_row))
                {
                    if allowed_walls[(dir, pos)] == WallType::Pocket {
                        // We only need to exclude this pocket..
                    }
                    self.exclude_squares(
                        &new_board.open_routes,
                        &distances_to_finish.dist,
                        new_pawn,
                    );
                    self
                } else {
                    for position in positions_to_check_for_excluding(dir, pos) {
                        if self.squares[position] == Some(-1) || self.squares[position] == None {
                            continue;
                        }
                        self.exclude_from_position(
                            position,
                            new_pawn,
                            &distances_to_finish.dist,
                            &new_board.open_routes,
                        );
                    }
                    self
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DistancesToFinish {
    pub dist: [[Option<i8>; 9]; 9],
}

impl DistancesToFinish {
    pub const fn zero() -> Self {
        Self {
            dist: [[None; 9]; 9],
        }
    }
    pub fn wall_parrallel(&self, dir: WallDirection, pos: Position) -> bool {
        let cross = positions_across_wall(dir, pos);
        self.dist[cross[0][0]] == self.dist[cross[0][1]]
            && self.dist[cross[1][0]] == self.dist[cross[1][1]]
    }

    // The pawn is in on the closer side to the finish of this wall
    pub fn pawn_short_side_wall(
        &self,
        wall_dir: WallDirection,
        wall_pos: Position,
        pawn_pos: Position,
    ) -> bool {
        let closest_dist = closest_square_distance(wall_pos, &self.dist);
        if closest_dist == self.dist[pawn_pos] {
            return true;
        }
        let _comp_row = pawn_pos.row;
        let _comp_col = pawn_pos.col;
        let add_to_index = match wall_dir {
            WallDirection::Horizontal => {
                if wall_pos.row == pawn_pos.row {
                    0
                } else {
                    1
                }
            }
            WallDirection::Vertical => {
                if wall_pos.col == pawn_pos.col {
                    0
                } else {
                    1
                }
            }
        };
        let cross = positions_across_wall(wall_dir, wall_pos);
        self.dist[cross[0][(0 + add_to_index) % 2]] == self.dist[cross[0][(1 + add_to_index) % 2]]
            && self.dist[cross[1][(0 + add_to_index) % 2]]
                == self.dist[cross[1][(1 + add_to_index) % 2]]
    }

    // With a pawn_move distances to finish stay the same.
    pub fn calculate_new_cache(self, game_move: Move, pawn_index: usize, board: &Board) -> Self {
        match game_move {
            Move::PawnMove(_, _) => self,
            Move::Wall(dir, pos) => {
                // If the wall is parrallel to the paths to the finish, then we don't need to calculate the distances_to_finish again.
                if self.wall_parrallel(dir, pos) {
                    return self;
                }
                // Now if its not parrallel do we actually want to calculate it?
                // We could try doing that by first removing all distances that are potentially affected. And then recalculating.
                // This would probably be quicker then just recalculating for scenarios where the wall doesn't affect that many tiles.
                // TODO !!!!!
                board.distance_to_finish_line(pawn_index)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NextMovesCache {
    pub relevant_squares: RelevantSquares,
    pub allowed_walls_for_pawn: AllowedWalls,
    pub distances_to_finish: DistancesToFinish,
}

impl NextMovesCache {
    pub const fn zero() -> Self {
        Self {
            relevant_squares: RelevantSquares::zero(),
            allowed_walls_for_pawn: AllowedWalls::zero(),
            distances_to_finish: DistancesToFinish::zero(),
        }
    }
    pub fn new(board: &Board, pawn_index: usize) -> Self {
        let pawn = board.pawns[pawn_index];

        let distances_to_finish = board.distance_to_finish_line(pawn_index);
        let (allowed_walls_for_pawn, wall_effects) =
            board.allowed_walls_for_pawn(pawn, &distances_to_finish);

        let relevant_squares = RelevantSquares::new(
            board,
            pawn_index,
            &distances_to_finish,
            &allowed_walls_for_pawn,
        );
        let allowed_walls_with_score =
            wall_effects.new_allowed_with_score(relevant_squares, allowed_walls_for_pawn);

        Self {
            relevant_squares,
            allowed_walls_for_pawn: allowed_walls_with_score,
            distances_to_finish,
        }
    }
    pub fn next_cache(
        &self,
        game_move: Move,
        old_board: &Board,
        new_board: &Board,
        pawn_index: usize,
    ) -> Self {
        let is_this_cache_turn = old_board.turn % 2 == pawn_index;
        let new_distances = self
            .distances_to_finish
            .calculate_new_cache(game_move, pawn_index, new_board);
        let (new_allowed_walls, wall_effects) = self.allowed_walls_for_pawn.calculate_new_cache(
            game_move,
            old_board.pawns[pawn_index],
            is_this_cache_turn,
            new_board,
            &new_distances,
        );
        let new_relevant_squares = self.relevant_squares.new_cache(
            game_move,
            new_board,
            old_board.pawns[pawn_index],
            pawn_index,
            is_this_cache_turn,
            &new_distances,
            &new_allowed_walls,
        );

        let allowed_walls_with_score =
            wall_effects.new_allowed_with_score(new_relevant_squares, new_allowed_walls);
        Self {
            relevant_squares: new_relevant_squares,
            allowed_walls_for_pawn: allowed_walls_with_score,
            distances_to_finish: new_distances,
        }
    }
}

// We only allow walls if they are allowed for both pawns
pub fn allowed_walls(next_moves_cache: &[NextMovesCache; 2]) -> Walls {
    let mut allowed_walls = Walls::new_allowed();
    for wall_dir in [WallDirection::Horizontal, WallDirection::Vertical] {
        for row in 0..8 {
            for col in 0..8 {
                let pos = Position { row, col };
                let wall = (wall_dir, pos);
                allowed_walls[wall] = next_moves_cache[0].allowed_walls_for_pawn[wall].is_allowed()
                    && next_moves_cache[1].allowed_walls_for_pawn[wall].is_allowed();
            }
        }
    }
    allowed_walls
}

fn positions_same_side_of_wall(dir: WallDirection, pos: Position) -> [[Position; 2]; 2] {
    match dir {
        WallDirection::Horizontal => [
            [
                Position {
                    row: pos.row,
                    col: pos.col,
                },
                Position {
                    row: pos.row,
                    col: pos.col + 1,
                },
            ],
            [
                Position {
                    row: pos.row + 1,
                    col: pos.col,
                },
                Position {
                    row: pos.row + 1,
                    col: pos.col + 1,
                },
            ],
        ],
        WallDirection::Vertical => [
            [
                Position {
                    row: pos.row,
                    col: pos.col,
                },
                Position {
                    row: pos.row + 1,
                    col: pos.col,
                },
            ],
            [
                Position {
                    row: pos.row,
                    col: pos.col + 1,
                },
                Position {
                    row: pos.row + 1,
                    col: pos.col + 1,
                },
            ],
        ],
    }
}

// Returns the four positions that are across from each other over the wall in pairs of two.
fn positions_across_wall(dir: WallDirection, pos: Position) -> [[Position; 2]; 2] {
    match dir {
        WallDirection::Horizontal => [
            [
                Position {
                    row: pos.row,
                    col: pos.col,
                },
                Position {
                    row: pos.row + 1,
                    col: pos.col,
                },
            ],
            [
                Position {
                    row: pos.row,
                    col: pos.col + 1,
                },
                Position {
                    row: pos.row + 1,
                    col: pos.col + 1,
                },
            ],
        ],
        WallDirection::Vertical => [
            [
                Position {
                    row: pos.row,
                    col: pos.col,
                },
                Position {
                    row: pos.row,
                    col: pos.col + 1,
                },
            ],
            [
                Position {
                    row: pos.row + 1,
                    col: pos.col,
                },
                Position {
                    row: pos.row + 1,
                    col: pos.col + 1,
                },
            ],
        ],
    }
}

fn wall_across_path(
    mut old_pos: Position,
    pawn_move: PawnMove,
    second_move: Option<PawnMove>,
) -> Vec<(WallDirection, Position)> {
    let mut walls_across_path = vec![];
    add_walls_accross_path(old_pos, pawn_move, &mut walls_across_path);
    old_pos = old_pos.add_move(pawn_move);
    if let Some(pawn_move) = second_move {
        add_walls_accross_path(old_pos, pawn_move, &mut walls_across_path);
    }
    walls_across_path
}
fn add_walls_accross_path(
    old_pos: Position,
    pawn_move: PawnMove,
    walls_across_path: &mut Vec<(WallDirection, Position)>,
) {
    let wall_direction = pawn_move.orthogonal_walls();
    let Position { row, col } = old_pos;
    match pawn_move {
        PawnMove::Down | PawnMove::Up => {
            let row = if pawn_move == PawnMove::Up {
                row
            } else {
                row - 1
            };
            if col <= 7 {
                walls_across_path.push((wall_direction, Position { row, col: col }));
            }
            if col >= 1 {
                walls_across_path.push((wall_direction, Position { row, col: col - 1 }));
            }
        }
        PawnMove::Left | PawnMove::Right => {
            let col = if pawn_move == PawnMove::Right {
                col
            } else {
                col - 1
            };
            if row <= 7 {
                walls_across_path.push((wall_direction, Position { row: row, col: col }));
            }
            if row >= 1 {
                walls_across_path.push((
                    wall_direction,
                    Position {
                        row: row - 1,
                        col: col,
                    },
                ));
            }
        }
    };
}

fn positions_to_check_for_excluding(dir: WallDirection, loc: Position) -> Vec<Position> {
    let mut to_check = vec![];
    match dir {
        WallDirection::Horizontal => {
            for row_add in 0..2 {
                for col_add in 0..4 {
                    let row = loc.row + row_add;
                    let col = loc.col - 1 + col_add;
                    if row < 0 || row > 8 || col < 0 || col > 8 {
                        continue;
                    }
                    to_check.push(Position { row, col });
                }
            }
        }
        WallDirection::Vertical => {
            for row_add in 0..4 {
                for col_add in 0..2 {
                    let row = loc.row - 1 + row_add;
                    let col = loc.col + col_add;
                    if row < 0 || row > 8 || col < 0 || col > 8 {
                        continue;
                    }
                    to_check.push(Position { row, col });
                }
            }
        }
    };
    to_check
}

// Return the minimum distance next to this wall, if there is no distance next to this wall, return None.
fn closest_square_distance(
    wall_location: Position,
    square_distances: &[[Option<i8>; 9]; 9],
) -> Option<i8> {
    let Position { row, col } = wall_location;
    let row = row as usize;
    let col = col as usize;
    let mut min_distance = None;
    for (row, col) in [
        (row, col),
        (row + 1, col),
        (row, col + 1),
        (row + 1, col + 1),
    ]
    .iter()
    {
        if let Some(distance) = square_distances[*row][*col] {
            if distance == -1 {
                // a -1 distance square is the same as a None distance square. Except we need it for walked unhindered
                continue;
            }
            if min_distance.is_none() || distance < min_distance.unwrap() {
                min_distance = Some(distance);
            }
        }
    }
    min_distance
}

fn closest_square_sum_distance(
    wall_location: Position,
    square_distances_0: &[[Option<i8>; 9]; 9],
    square_distances_1: &[[Option<i8>; 9]; 9],
) -> Option<i8> {
    let Position { row, col } = wall_location;
    let row = row as usize;
    let col = col as usize;
    let mut min_distance = None;
    for (row, col) in [
        (row, col),
        (row + 1, col),
        (row, col + 1),
        (row + 1, col + 1),
    ]
    .iter()
    {
        if let Some(distance_0) = square_distances_0[*row][*col] {
            if let Some(distance_1) = square_distances_1[*row][*col] {
                if distance_0 == -1 || distance_1 == -1 {
                    continue;
                }
                let distance = distance_0 + distance_1;
                if min_distance.is_none() || distance < min_distance.unwrap() {
                    min_distance = Some(distance);
                }
            }
        }
    }
    min_distance
}

#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpenRoutes {
    left_right: [[bool; 8]; 9],
    down_up: [[bool; 9]; 8],
}

impl OpenRoutes {
    fn new() -> Self {
        Self {
            left_right: [[true; 8]; 9],
            down_up: [[true; 9]; 8],
        }
    }
    pub fn is_open(&self, position: Position, pawn_move: PawnMove) -> bool {
        let Position { col, row } = position;
        let col = col as usize;
        let row = row as usize;
        let res = match pawn_move {
            PawnMove::Up => row < 8 && self.down_up[row][col],
            PawnMove::Down => row >= 1 && self.down_up[row - 1][col],
            PawnMove::Right => col < 8 && self.left_right[row][col],
            PawnMove::Left => col >= 1 && self.left_right[row][col - 1],
        };
        res
    }

    /// Should update the open path ways, The wall locations are inbetween the open path locations, so a horizontal wall at location 0,0 blocks the down_up at (0,0) and (1,0)
    fn update_open(&mut self, direction: WallDirection, location: Position) {
        let Position { row, col } = location;
        let row = row as usize;
        let col = col as usize;
        match direction {
            WallDirection::Horizontal => {
                self.down_up[row][col] = false;
                self.down_up[row][col + 1] = false;
            }
            WallDirection::Vertical => {
                self.left_right[row][col] = false;
                self.left_right[row + 1][col] = false;
            }
        }
    }

    // Use distances for this...
    pub fn shortest_path_from_dest_row_to_pawn(&self, pawn: Pawn) -> Vec<(PawnMove, Position)> {
        let mut previous_tile: [[Option<Position>; 9]; 9] = [[None; 9]; 9];
        let mut queue: ArrayDeque<Position, 40> = ArrayDeque::new();
        //let mut queue: VecDeque<Position> = VecDeque::new();
        queue.push_back(pawn.position).unwrap();
        previous_tile[pawn.position.row as usize][pawn.position.col as usize] = Some(pawn.position);
        let moves_order = if pawn.goal_row == 8 {
            PAWN_MOVES_DOWN_LAST
        } else {
            PAWN_MOVES_UP_LAST
        };
        let mut latest = pawn.position;
        'outer: while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            for pawn_move in moves_order {
                if self.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);
                    let old_value = &mut previous_tile[next.row as usize][next.col as usize];
                    if *old_value == None {
                        queue.push_back(next).unwrap();
                        *old_value = Some(current);
                    }
                    if next.row == pawn.goal_row {
                        latest = next;
                        break 'outer;
                    }
                }
            }
        }
        let mut shortest_path = vec![];
        let mut current = latest;
        while current != pawn.position {
            let previous = previous_tile[current.row as usize][current.col as usize].unwrap();
            shortest_path.push((current.move_to(previous).unwrap(), current));
            current = previous;
        }
        return shortest_path;
    }

    // We check if only one of its neighbours has a smaller distance to the pawn, otherwise we won't need to check it, (cause can not be excluded).
    //If in this dijkstra search we hit a square with distance from finish <= distance from distance at the start square.
    // We stop the search. Also if we hit a square where distance to pawn <= distance to pawn start we stop the search.
    // If the dijkstra search never hits a stop condition, we can exclude all squares hit with the dijkstra search.
    pub fn dijkstra_exclude_squares(
        &self,
        start_position: Position,
        distances_from_pawn: &[[Option<i8>; 9]; 9],
        distances_from_finish: &[[Option<i8>; 9]; 9],
    ) -> Option<Vec<Position>> {
        let moves_order = PAWN_MOVES_DOWN_LAST;
        let mut number_direct_neighbour_smaller = 0;
        // We could do with a question mark, but thats probably less clear.
        if distances_from_pawn[start_position].is_none() {
            return None;
        }
        // First we will check if for only one neighbour the distance to the pawn is smaller. If not we not exclude this square.
        for pawn_move in moves_order {
            if self.is_open(start_position, pawn_move) {
                let next = start_position.add_move(pawn_move);
                if distances_from_pawn[next].is_some()
                    && distances_from_pawn[next] < distances_from_pawn[start_position]
                {
                    number_direct_neighbour_smaller += 1;
                }
            }
        }
        let mut to_exclude = vec![];

        if number_direct_neighbour_smaller > 1 {
            return None;
        }

        let mut distances: [[Option<i8>; 9]; 9] = [[None; 9]; 9];
        let mut queue: ArrayDeque<Position, 40> = ArrayDeque::new();
        //let mut queue: VecDeque<Position> = VecDeque::new();
        queue.push_back(start_position).unwrap();
        to_exclude.push(start_position);
        distances[start_position] = Some(0);

        while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            let current_distance = distances[current].unwrap();
            'inner: for pawn_move in moves_order {
                if self.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);
                    // Here we do hit none. For previous excluded tiles. How to fix this.
                    if distances_from_pawn[next].is_some()
                        && distances_from_pawn[next] != Some(-1)
                        && distances_from_pawn[next] < distances_from_pawn[start_position]
                    {
                        if current_distance > 0 {
                            // We can get to a closer distance, so not closed off
                            return None;
                        }
                        // This continue is just for the starting position.
                        continue 'inner;
                    }
                    // whole edge case... necessary for last test case
                    // what to do if distances_from_pawn[start_position] == 0;

                    if distances_from_finish[next] < distances_from_finish[start_position] {
                        // We found a spot that is closer to the finish line, so there must be another route there. Hence the pawn could be forced back here.
                        return None;
                    }
                    if distances_from_pawn[next].is_none() || distances_from_pawn[next] == Some(-1)
                    {
                        continue;
                    }
                    let distance = &mut distances[next];
                    if distance.is_none() {
                        let next_distance = Some(current_distance + 1);
                        *distance = next_distance;
                        queue.push_back(next).unwrap();
                        to_exclude.push(next);
                    }
                }
            }
        }
        Some(to_exclude)
    }

    // In this function, we will systematically exclude squares, by checking if the pawn can be forced there.
    // To do this we wil loop through all the squares in distances from pawn, starting from the furthest away one.
    // We check if only one of its neighbours has a smaller distance, otherwise we won't need to check it, (cause can not be excluded).

    // Then we will start a dijkstra search from the square. If in this dijkstra search we hit a square with distance from finish <= distance from distance at the start square.
    // We stop the search. Also if we hit a square where distance to pawn <= distance to pawn start we stop the search.
    // If the dijkstra search never hits a stop condition, we can exclude all squares hit with the dijkstra search.
    pub fn exclude_squares(
        &self,
        distances_from_pawn: &mut [[Option<i8>; 9]; 9],
        distances_from_finish: &[[Option<i8>; 9]; 9],
    ) -> i8 {
        // we will loop through the distances from pawn, starting from far away to close by.
        let mut distances = vec![];
        let mut squares_excluded = 0;
        for row in 0..9 {
            for col in 0..9 {
                let position = Position { row, col };
                let distance = &distances_from_pawn[position];
                if let Some(distance) = distance {
                    distances.push((position, *distance));
                }
            }
        }
        distances.sort_by_key(|x| -x.1);
        for position in distances {
            if let Some(to_exclude) = self.dijkstra_exclude_squares(
                position.0,
                distances_from_pawn,
                distances_from_finish,
            ) {
                squares_excluded += to_exclude.len() as i8;
                for position in to_exclude {
                    distances_from_pawn[position] = None;
                }
            }
        }
        squares_excluded
    }

    // Here return all squares that this pawn can reach. And the shortest path to reach them.
    pub fn find_all_squares_relevant_for_pawn(
        &self,
        pawn: Pawn,
        allowed_walls: &AllowedWalls,
        distance_walked: i8,
    ) -> (i8, [[Option<i8>; 9]; 9]) {
        let mut distance: [[Option<i8>; 9]; 9] = [[None; 9]; 9];
        let mut queue: ArrayDeque<Position, 40> = ArrayDeque::new();
        //let mut queue: VecDeque<Position> = VecDeque::new();
        queue.push_back(pawn.position).unwrap();
        distance[pawn.position] = Some(distance_walked);
        let mut count = 1;
        let moves_order = if pawn.goal_row == 8 {
            PAWN_MOVES_DOWN_LAST
        } else {
            PAWN_MOVES_UP_LAST
        };
        'outer: while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            let current_distance = distance[current];
            'inner: for pawn_move in moves_order {
                if self.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);
                    if next.row == pawn.goal_row {
                        // If this path path to the goal line can't be blocked, The other pawn_moves are not relevant. Cause pawn can never be forced to go there.
                        for relevant_wall in current.relevant_walls(pawn_move) {
                            if allowed_walls[relevant_wall].is_allowed() {
                                continue 'inner;
                            }
                        }
                        // THis skip makes distances incorrect..
                        // But we update the distances latery anyway, so doesn't matter.
                        continue 'outer;
                    }

                    let distance = &mut distance[next];
                    if let Some(square_distance) = distance {
                        if *square_distance <= current_distance.unwrap() + 1 {
                            continue;
                        }
                    }
                    count += 1;
                    let next_distance = current_distance.map(|x| x + 1);
                    *distance = next_distance;
                    queue.push_back(next).unwrap();
                }
            }
        }
        // Now we want to think about what else we can conclude. For example a pawn can never be forced back up a one depth lane.
        // a position is part of one breath lane, if it has only 2 paths open to it. d
        (count, distance)
    }

    pub fn furthest_walkable_unhindered(
        &self,
        pawn: Pawn,
        allowed_walls: &AllowedWalls,
        distances_to_finish: &DistancesToFinish,
    ) -> Vec<Position> {
        let mut walked_to = [[None; 9]; 9];
        let mut finished_paths = vec![];
        let mut best_to_explore: BinaryHeap<(usize, Position)> = BinaryHeap::new();
        best_to_explore.push((1, pawn.position));
        while let Some((number_of_steps, current)) = best_to_explore.pop() {
            let mut made_step = false;
            'inner: for pawn_move in PAWN_MOVES_DOWN_LAST {
                if self.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);
                    let current_distance_to_finish = distances_to_finish.dist[current].unwrap();

                    if distances_to_finish.dist[next].unwrap() >= current_distance_to_finish {
                        continue 'inner;
                    }
                    // This path joins an already walked to path, so irrelevant
                    if walked_to[next].is_some() {
                        continue 'inner;
                    }

                    // If a wall can be placed, we break the loop.
                    for relevant_walls in current.relevant_walls(pawn_move) {
                        let wall_type = allowed_walls[relevant_walls];
                        match wall_type {
                            WallType::Allowed(_) | WallType::Pocket => {
                                continue 'inner;
                            }
                            WallType::Impossible => (),
                            WallType::Unallowed => {
                                let mut pawn = pawn;
                                pawn.position = current;
                                let mut open_routes = self.clone();
                                open_routes.update_open(relevant_walls.0, relevant_walls.1);
                                if open_routes
                                    .find_path_for_pawn_to_dest_row(pawn, false)
                                    .is_some()
                                {
                                    // wall  still not allowed
                                    continue 'inner;
                                }
                            }
                        };
                    }
                    best_to_explore.push((number_of_steps + 1, next));
                    walked_to[next] = Some(current);
                    made_step = true;
                }
            }
            if !made_step {
                finished_paths.push(Self::follow_path_back(current, &walked_to));
            }
        }
        let mut longest_path = vec![];
        for path in finished_paths {
            if path.len() >= longest_path.len() {
                longest_path = path;
            }
        }
        longest_path
    }

    fn follow_path_back(
        position: Position,
        prev_step: &[[Option<Position>; 9]; 9],
    ) -> Vec<Position> {
        let mut steps_back = vec![];
        let mut current = position;
        while let Some(next) = prev_step[current] {
            steps_back.push(current);
            current = next;
        }
        steps_back.push(current);
        steps_back.reverse();
        steps_back
    }
    // Takes all the steps for this pawn along the shortest path, as long as the path can never be blocked off.
    // Returns the number of steps taken and the new position
    // Contains an error currently. If we walk are in tunnel, but back is till open, it will take one step forward.
    // TODO: WHAT TO DO IF THERE ARE 2 STEPS THAT ARE NOT BLOCKED? We will have to explore both of them I Guess?
    pub fn pawn_walk_unhindered(
        &self,
        pawn: Pawn,
        allowed_walls: &AllowedWalls,
        distances_to_finish: &DistancesToFinish,
    ) -> Option<(Vec<Position>, [[Option<i8>; 9]; 9])> {
        let mut to_return =
            self.furthest_walkable_unhindered(pawn, allowed_walls, distances_to_finish);
        //        let mut current_distance_to_finish = distances_to_finish[pawn.position].unwrap();
        //        let mut current = pawn.position;
        //        let mut to_return = vec![pawn.position];
        //        let mut made_step = true;
        //        while made_step {
        //            made_step = false;
        //            'inner: for pawn_move in PAWN_MOVES_DOWN_LAST {
        //                if self.is_open(current, pawn_move) {
        //                    let next = current.add_move(pawn_move);
        //                    if distances_to_finish[next].unwrap() >= current_distance_to_finish {
        //                        continue 'inner;
        //                    }
        //                    // If a wall can be placed, we break the loop.
        //                    for relevant_walls in current.relevant_walls(pawn_move) {
        //                        let wall_type = allowed_walls[relevant_walls];
        //                        match wall_type {
        //                            WallType::Allowed | WallType::Pocket => {
        //                                continue 'inner;
        //                            }
        //                            WallType::Impossible => (),
        //                            WallType::Unallowed => {
        //                                let mut pawn = pawn;
        //                                pawn.position = current;
        //                                let mut open_routes = self.clone();
        //                                open_routes.update_open(relevant_walls.0, relevant_walls.1);
        //                                if open_routes
        //                                    .find_path_for_pawn_to_dest_row(pawn, false)
        //                                    .is_some()
        //                                {
        //                                    // wall  still not allowed
        //                                    continue 'inner;
        //                                }
        //                            }
        //                        };
        //                    }
        //                    current_distance_to_finish = distances_to_finish[next].unwrap();
        //                    current = next;
        //                    to_return.push(next);
        //                    made_step = true;
        //                    break 'inner;
        //                }
        //            }
        //        }
        if to_return.len() == 1 {
            return None;
        }
        let mut previous_pos = to_return.last().unwrap();
        let mut walked_to_unhindered = 1;
        let mut pocket_response = None;
        if to_return.last().unwrap().row == pawn.goal_row {
            return Some((to_return, [[Some(2); 9]; 9]));
        };
        for (i, pos) in to_return.iter().rev().skip(1).enumerate() {
            let first_move = previous_pos.substract_pos(pos).unwrap();
            let mut pawn = pawn;
            pawn.position = *previous_pos;
            // This below goes wrong, cause the pawn is always on the line....
            match self.check_if_line_only_entrance_v1(
                [*previous_pos; MAX_LINE_LENGTH],
                pawn,
                first_move,
                &distances_to_finish.dist,
            ) {
                PocketCheckResponse::NoPocket(_) => (),
                PocketCheckResponse::Pocket(pocket_resp) => {
                    // Now we want to exlude all of these squares
                    walked_to_unhindered = to_return.len() - i;
                    pocket_response = Some(pocket_resp);
                    break;
                }
            };
            previous_pos = pos;
        }
        let pocket_resp = pocket_response?;
        to_return.truncate(walked_to_unhindered);
        Some((to_return, pocket_resp))
    }

    pub fn test_check_lines(
        &self,
        pawn: Pawn,
        relevant_squares: &RelevantSquares,
        distances_to_finish: DistancesToFinish,
    ) -> [[i8; 9]; 9] {
        let mut blocked_off: [[i8; 9]; 9] = [[0; 9]; 9];
        for row in 0..9 {
            for col in 0..9 {
                let pos = Position { row, col };
                for dir in [PawnMove::Right, PawnMove::Up] {
                    if let Some((line_length, pocket)) = self.check_blocked_off(
                        pos,
                        dir,
                        pawn,
                        relevant_squares,
                        &distances_to_finish.dist,
                    ) {
                        let line_score = if line_length <= 3 { 2 } else { 1 };
                        for row in 0..9 {
                            for col in 0..9 {
                                let pos = Position { row, col };
                                if pocket[pos].is_some() {
                                    blocked_off[pos] += line_score;
                                }
                            }
                        }
                    }
                }
            }
        }
        blocked_off
    }

    pub fn check_blocked_off_by_line(
        &self,
        line: (u8, [Position; MAX_LINE_LENGTH]),
        line_direction: WallDirection,
        pawn: Pawn,
        relevant_squares: &RelevantSquares,
        distances_to_finish: &[[Option<i8>; 9]; 9],
    ) -> Option<(u8, [[Option<i8>; 9]; 9])> {
        if !self.check_if_line_potential_pocket(line.1, line_direction, relevant_squares) {
            return None;
        }
        let directions_to_check = match line_direction {
            WallDirection::Horizontal => [PawnMove::Up, PawnMove::Down],
            WallDirection::Vertical => [PawnMove::Left, PawnMove::Right],
        };
        for direction in directions_to_check {
            // TODO add performance increase, if we know that we hit all the squares one way around, we don't need to check the other way around.
            // Cause that will never be a pocket
            match self.check_if_line_only_entrance_v1(line.1, pawn, direction, distances_to_finish)
            {
                PocketCheckResponse::NoPocket(false) => continue,
                PocketCheckResponse::NoPocket(true) => return None,
                PocketCheckResponse::Pocket(distance) => return Some((line.0, distance)),
            }
        }
        None
    }

    pub fn check_blocked_off(
        &self,
        left_corner: Position,
        direction: PawnMove,
        pawn: Pawn,
        relevant_squares: &RelevantSquares,
        distances_to_finish: &[[Option<i8>; 9]; 9],
    ) -> Option<(u8, [[Option<i8>; 9]; 9])> {
        if relevant_squares.squares[left_corner].is_none()
            || relevant_squares.squares[left_corner].unwrap() == -1
        {
            return None;
        }
        let line = self.create_line(left_corner, direction)?;
        let line_direction = match direction {
            PawnMove::Left | PawnMove::Right => WallDirection::Horizontal,
            PawnMove::Up | PawnMove::Down => WallDirection::Vertical,
        };
        self.check_blocked_off_by_line(
            line,
            line_direction,
            pawn,
            relevant_squares,
            distances_to_finish,
        )
    }

    // Here we create a line of 2 or 3 positions, depending on how many can be connected
    pub fn create_line(
        &self,
        left_corner: Position,
        direction: PawnMove,
    ) -> Option<(u8, [Position; MAX_LINE_LENGTH])> {
        for pawn_move in direction.orthogonal_moves() {
            if !self.is_open(left_corner, pawn_move) {
                return None;
            }
        }

        let mut length = 1;
        let mut current = left_corner;
        let mut line = [left_corner; MAX_LINE_LENGTH];
        'outer: for i in 0..(MAX_LINE_LENGTH - 1) {
            if self.is_open(current, direction) {
                current = current.add_move(direction);
                for pawn_move in direction.orthogonal_moves() {
                    if !self.is_open(current, pawn_move) {
                        break 'outer;
                    }
                }

                length += 1;
                line[i + 1] = current;
            } else {
                break;
            }
        }
        line[MAX_LINE_LENGTH - 1] = current;
        if length <= 1 {
            return None;
        }
        Some((length, line))
    }

    pub fn wall_next_to_position(&self, pos: Position, relevant_direction: PawnMove) -> bool {
        if !self.is_open(pos, relevant_direction) {
            return true;
        }
        let next = pos.add_move(relevant_direction);
        match relevant_direction {
            PawnMove::Up | PawnMove::Down => {
                if !self.is_open(next, PawnMove::Left) || !self.is_open(next, PawnMove::Right) {
                    return true;
                }
            }
            PawnMove::Left | PawnMove::Right => {
                if !self.is_open(next, PawnMove::Up) || !self.is_open(next, PawnMove::Down) {
                    return true;
                }
            }
        }

        false
    }

    // A line is only a potential pocket if it is surrounded by two walls.
    // either orthogonal/parrallel walls on both sides.
    pub fn check_if_line_potential_pocket(
        &self,
        line: [Position; MAX_LINE_LENGTH],
        line_direction: WallDirection,
        relevant_squares: &RelevantSquares,
    ) -> bool {
        for pos in line {
            if relevant_squares.squares[pos].is_none()
                || relevant_squares.squares[pos].unwrap() == -1
                || relevant_squares.squares[pos].unwrap() == 0
            {
                return false;
            }
        }
        match line_direction {
            WallDirection::Horizontal => {
                let left = self.wall_next_to_position(line[0], PawnMove::Left);
                let right = self.wall_next_to_position(line[MAX_LINE_LENGTH - 1], PawnMove::Right);
                left && right
            }
            WallDirection::Vertical => {
                let up = self.wall_next_to_position(line[0], PawnMove::Down);
                let down = self.wall_next_to_position(line[MAX_LINE_LENGTH - 1], PawnMove::Up);
                up && down
            }
        }
    }
    pub fn check_if_line_only_entrance_v1(
        &self,
        line: [Position; MAX_LINE_LENGTH],
        pawn: Pawn,
        first_move: PawnMove,
        distances_from_finish: &[[Option<i8>; 9]; 9],
    ) -> PocketCheckResponse {
        let mut distance: [[Option<i8>; 9]; 9] = [[None; 9]; 9];
        let mut queue: ArrayDeque<Position, 40> = ArrayDeque::new();
        //let mut queue: VecDeque<Position> = VecDeque::new();
        //println!("positions: {:?}", positions);
        //println!("other_side {:?}", other_side);
        let mut min_finish_distance = distances_from_finish[line[0]].unwrap();
        for i in 1..MAX_LINE_LENGTH {
            min_finish_distance = min_finish_distance.min(distances_from_finish[line[i]].unwrap());
        }

        for position in line {
            if self.is_open(position, first_move) {
                let next = position.add_move(first_move);
                if next.row == pawn.goal_row || next == pawn.position {
                    return PocketCheckResponse::NoPocket(false);
                }
                queue.push_back(next).unwrap();
                distance[next] = Some(0);
            }
        }

        let moves_order = PAWN_MOVES_UP_LAST;
        while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            let current_distance = distance[current];
            for pawn_move in moves_order {
                if self.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);
                    if line.contains(&next) {
                        if pawn_move == first_move {
                            // We reached the line from the other way round.
                            return PocketCheckResponse::NoPocket(true);
                        } else {
                            continue;
                        }
                    }

                    if next.row == pawn.goal_row || next == pawn.position {
                        return PocketCheckResponse::NoPocket(false);
                    }
                    if distances_from_finish[next].unwrap() < min_finish_distance {
                        return PocketCheckResponse::NoPocket(false);
                    }

                    let distance = &mut distance[next];
                    if distance.is_none() {
                        let next_distance = current_distance.map(|x| x + 1);
                        *distance = next_distance;
                        queue.push_back(next).unwrap();
                    }
                }
            }
        }
        return PocketCheckResponse::Pocket(distance);
    }

    // Here we will check if this line is the only entrance into this area of the board, and if this area contains neither the pawn/ goal_row of this pawn.
    // In that case that area is less relevant to consider.
    pub fn check_if_line_only_entrance(
        &self,
        line: [Position; MAX_LINE_LENGTH],
        pawn: Pawn,
        first_move: PawnMove,
        relevant_squares: &RelevantSquares,
        distances_from_finish: &[[Option<i8>; 9]; 9],
    ) -> PocketCheckResponse {
        let mut distance: [[Option<i8>; 9]; 9] = [[None; 9]; 9];
        let mut queue: ArrayDeque<Position, 40> = ArrayDeque::new();
        //let mut queue: VecDeque<Position> = VecDeque::new();
        //println!("positions: {:?}", positions);
        //println!("other_side {:?}", other_side);

        let min_pawn_distance = relevant_squares.squares[line[0]].unwrap();
        let min_finish_distance = distances_from_finish[line[0]].unwrap();

        for position in line {
            if self.is_open(position, first_move) {
                let next = position.add_move(first_move);
                if next.row == pawn.goal_row || next == pawn.position {
                    return PocketCheckResponse::NoPocket(false);
                }
                queue.push_back(next).unwrap();
                distance[next] = Some(0);
            }
        }

        let moves_order = PAWN_MOVES_UP_LAST;
        while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            let current_distance = distance[current];
            for pawn_move in moves_order {
                if self.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);
                    if line.contains(&next) {
                        if pawn_move == first_move {
                            // We reached the line from the other way round.
                            return PocketCheckResponse::NoPocket(true);
                        } else {
                            continue;
                        }
                    }
                    if next.row == pawn.goal_row || next == pawn.position {
                        return PocketCheckResponse::NoPocket(false);
                    }
                    if relevant_squares.squares[next].is_none()
                        || relevant_squares.squares[next].unwrap() == -1
                    {
                        continue;
                    }
                    if relevant_squares.squares[next].unwrap() < min_pawn_distance
                        || distances_from_finish[next].unwrap() < min_finish_distance
                    {
                        // We reached a square that is closer to the pawn/goal_row, so this line is not the only entrance.
                        return PocketCheckResponse::NoPocket(false);
                    }

                    let distance = &mut distance[next];
                    if distance.is_none() {
                        let next_distance = current_distance.map(|x| x + 1);
                        *distance = next_distance;
                        queue.push_back(next).unwrap();
                    }
                }
            }
        }
        return PocketCheckResponse::Pocket(distance);
    }

    // Here we check if this wall blocks off an area with neither the goal_row,
    // nor the position of this pawn. In that case the wall is less relevant to consider.
    pub fn check_if_pocket(
        &self,
        positions: [Position; 2],
        pawn: Pawn,
        other_side: [Position; 2],
        // returns: wall_allowed, is_pocket_0, is_pocket_1, reached_otherside
    ) -> CheckPocketResponseNew {
        let mut distance: [[Option<u8>; 9]; 9] = [[None; 9]; 9];
        let mut queue: ArrayDeque<Position, 40> = ArrayDeque::new();
        //let mut queue: VecDeque<Position> = VecDeque::new();
        //println!("positions: {:?}", positions);
        //println!("other_side {:?}", other_side);
        let mut goal_row_seen = false;
        let mut pawn_seen = false;

        for position in positions {
            queue.push_back(position).unwrap();
            distance[position] = Some(0);
            pawn_seen = pawn_seen || position == pawn.position;
            goal_row_seen = goal_row_seen || position.row == pawn.goal_row;
        }
        let moves_order = PAWN_MOVES_UP_LAST;
        let mut reached_otherside = None;
        while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            let current_distance = distance[current];
            for pawn_move in moves_order {
                if self.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);
                    goal_row_seen = goal_row_seen || next.row == pawn.goal_row;
                    pawn_seen = pawn_seen || next == pawn.position;
                    if other_side.contains(&next) {
                        // In case that we reached the otherside, the wall is always allowed, and never a pocket, so we can break quicker
                        reached_otherside = current_distance.map(|x| (x + 1) as i8);
                        break;
                    }
                    let distance = &mut distance[next];
                    if distance.is_none() {
                        let next_distance = current_distance.map(|x| x + 1);
                        *distance = next_distance;
                        queue.push_back(next).unwrap();
                    }
                }
            }
        }
        // TODO: To decide on a wall score, see how big the area that is blocked off is, or how many steps it takes to walk around a wall.
        CheckPocketResponseNew {
            wall_allowed: !(pawn_seen && !goal_row_seen) || reached_otherside.is_some(),
            is_pocket: !(goal_row_seen || pawn_seen || reached_otherside.is_some()),
            pawn_seen,
            goal_row_seen,
            reached_otherside,
            area_seen: distance,
        }
    }

    // Returns the path length of the shortest path for this pawn to the other side, using the dijkstra algorithm on the open_routes.
    pub fn find_path_for_pawn_to_dest_row(&self, pawn: Pawn, distance_exact: bool) -> Option<u8> {
        let mut distance: [[Option<u8>; 9]; 9] = [[None; 9]; 9];
        let mut queue: ArrayDeque<Position, 40> = ArrayDeque::new();
        //let mut queue: VecDeque<Position> = VecDeque::new();
        queue.push_back(pawn.position).unwrap();
        distance[pawn.position.row as usize][pawn.position.col as usize] = Some(0);
        let moves_order = if pawn.goal_row == 8 {
            PAWN_MOVES_UP_LAST
        } else {
            PAWN_MOVES_DOWN_LAST
        };
        while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            let current_distance = distance[current];
            for pawn_move in moves_order {
                if self.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);
                    if next.row == pawn.goal_row {
                        return current_distance.map(|x| x + 1);
                    }
                    let distance = &mut distance[next];
                    if distance.is_none() {
                        let next_distance = current_distance.map(|x| x + 1);
                        *distance = next_distance;
                        if distance_exact {
                            queue.push_back(next).unwrap();
                        } else {
                            queue.push_front(next).unwrap();
                        }
                    }
                }
            }
        }
        None
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum MoveResult {
    Valid,
    Invalid,
    Win,
}

impl MoveResult {
    pub fn player_moved(&self) -> PlayerMoveResult {
        match self {
            MoveResult::Valid => PlayerMoveResult::Valid,
            MoveResult::Invalid => PlayerMoveResult::Invalid,
            MoveResult::Win => PlayerMoveResult::Win,
        }
    }

    pub fn ai_moved(&self) -> PlayerMoveResult {
        match self {
            MoveResult::Valid => PlayerMoveResult::Valid,
            MoveResult::Invalid => PlayerMoveResult::Invalid,
            MoveResult::Win => PlayerMoveResult::Lose,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Board {
    pub turn: usize,
    pub pawns: [Pawn; 2],
    pub walls: Walls,
    allowed_walls: Walls,
    pub open_routes: OpenRoutes,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct MinimalBoard {
    pub turn: usize,
    pub pawns: [Position; 2],
    pub walls: Walls,
}

impl MinimalBoard {
    // this one is unchecked, so should be check by a full board
    fn game_move(&mut self, game_move: Move) {
        match game_move {
            Move::PawnMove(pawn_move, second_pawn_move) => {
                let pawn_index = self.turn % 2;
                self.pawns[pawn_index] = self.pawns[pawn_index].add_move(pawn_move);
                if let Some(second_pawn_move) = second_pawn_move {
                    self.pawns[pawn_index] = self.pawns[pawn_index].add_move(second_pawn_move);
                }
            }
            Move::Wall(direction, location) => {
                self.walls.place_wall(direction, location);
            }
        }
        self.turn = self.turn + 1;
    }
}

// The board has three public methods, new, place_wall and move_pawn.
impl Board {
    pub fn new() -> Self {
        Self {
            turn: 0,
            pawns: [Pawn::new_bottom(), Pawn::new_top()],
            walls: Walls::default(),
            allowed_walls: Walls::new_allowed(),
            open_routes: OpenRoutes::new(),
        }
    }

    pub fn encode(&self) -> String {
        let mut encoding = format!(
            "{};{};{}",
            self.turn,
            self.pawns[0].encode(),
            self.pawns[1].encode()
        );
        for row in 0..8 {
            for col in 0..8 {
                let pos = Position {
                    row: row as i8,
                    col: col as i8,
                };
                if self.walls.horizontal[row][col] {
                    encoding = format!("{};{}h", encoding, pos.encode());
                }
                if self.walls.vertical[row][col] {
                    encoding = format!("{};{}v", encoding, pos.encode());
                }
            }
        }

        encoding
    }

    pub fn decode(input: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut board = Board::new();
        let mut split = input.split(';');
        let turn = split
            .next()
            .ok_or("missing turn number")?
            .parse::<usize>()?;
        let pawn_0 = Pawn::decode(split.next().ok_or("missing pawn 0 data")?, 0)?;
        let pawn_1 = Pawn::decode(split.next().ok_or("missing pawn 1 data")?, 1)?;
        board.pawns[0].position = pawn_0.position;
        board.pawns[1].position = pawn_1.position;
        for wall in split {
            let pos = Position::decode(wall)?;
            let wall_direction = if wall.ends_with('h') {
                WallDirection::Horizontal
            } else if wall.ends_with('v') {
                WallDirection::Vertical
            } else {
                return Err("Wall direction not found")?;
            };
            if !board.place_wall(wall_direction, pos) {
                return Err("Wall placement failed, invalid board")?;
            }
        }
        board.turn = turn;
        board.pawns = [pawn_0, pawn_1];
        Ok(board)
    }

    // Returns which pawn is at a row and column, if any.
    pub fn is_pawn(&self, row: usize, col: usize) -> Option<usize> {
        for (i, pawn) in self.pawns.iter().enumerate() {
            if pawn.position.row as usize == row && pawn.position.col as usize == col {
                return Some(i);
            }
        }
        None
    }
    pub fn allowed_walls_for_pawn(
        &self,
        pawn: Pawn,
        distance_to_finish: &DistancesToFinish,
    ) -> (AllowedWalls, WallEffects) {
        let mut allowed_walls = AllowedWalls::new();
        let mut wall_effects = WallEffects::new();
        for row in 0..8 {
            for col in 0..8 {
                let location = Position {
                    row: row as i8,
                    col: col as i8,
                };
                for direction in [WallDirection::Horizontal, WallDirection::Vertical].iter() {
                    let wall = (*direction, location);
                    if !self.allowed_walls.is_allowed(*direction, location) {
                        allowed_walls[wall] = WallType::Impossible;
                    }
                    // Here we update walls that not allowed because of path blocking
                    let (allowed, effect) = self.is_wall_allowed_pawn_new(
                        (*direction, location),
                        pawn,
                        distance_to_finish,
                    );
                    allowed_walls[wall] = allowed;
                    wall_effects[wall] = effect;
                }
            }
        }
        (allowed_walls, wall_effects)
    }

    pub fn allowed_walls(&self) -> Walls {
        let mut allowed_walls = self.allowed_walls;
        for row in 0..8 {
            for col in 0..8 {
                let location = Position {
                    row: row as i8,
                    col: col as i8,
                };
                for direction in [WallDirection::Horizontal, WallDirection::Vertical].iter() {
                    if !allowed_walls.is_allowed(*direction, location) {
                        continue;
                    }
                    // Here we update walls that not allowed because of path blocking
                    if !self.is_wall_allowed(*direction, location) {
                        allowed_walls.wall_not_allowed(*direction, location);
                    }
                }
            }
        }
        allowed_walls
    }

    pub fn can_place_wall(&self, direction: WallDirection, location: Position) -> bool {
        if self.pawns[self.turn % 2].number_of_walls_left == 0 {
            return false;
        }
        self.is_wall_allowed(direction, location)
    }

    pub fn place_wall(&mut self, direction: WallDirection, location: Position) -> bool {
        if !self.can_place_wall(direction, location) {
            return false;
        }
        self.pawns[self.turn % 2].number_of_walls_left -= 1;
        self.turn = self.turn + 1;
        self.walls.place_wall(direction, location);
        self.allowed_walls.update_allowed(direction, location);
        self.open_routes.update_open(direction, location);
        true
    }

    pub fn can_move_pawn(
        &self,
        pawn_move: (PawnMove, Option<PawnMove>),
    ) -> (MoveResult, Option<Position>) {
        let pawn = self.pawns[self.turn % 2];
        let mut next = pawn.position.add_move(pawn_move.0);
        let valid_move = if self.open_routes.is_open(pawn.position, pawn_move.0) {
            if self.pawns[(self.turn + 1) % 2].position == next {
                // We need to jump, so two moves required
                if let Some(second_pawn_move) = pawn_move.1 {
                    if self.open_routes.is_open(next, pawn_move.0)
                        && second_pawn_move == pawn_move.0
                    {
                        next = next.add_move(second_pawn_move);
                        true
                    } else if !self.open_routes.is_open(next, pawn_move.0)
                        && pawn_move.0.orthogonal_moves().contains(&second_pawn_move)
                        && self.open_routes.is_open(next, second_pawn_move)
                    {
                        next = next.add_move(second_pawn_move);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                true
            }
        } else {
            false
        };
        let win = next.row == self.pawns[self.turn % 2].goal_row;
        if valid_move {
            if win {
                (MoveResult::Win, Some(next))
            } else {
                (MoveResult::Valid, Some(next))
            }
        } else {
            (MoveResult::Invalid, None)
        }
    }

    pub fn move_pawn(&mut self, pawn_move: (PawnMove, Option<PawnMove>)) -> MoveResult {
        let (move_result, next) = self.can_move_pawn(pawn_move);
        if let Some(next) = next {
            self.pawns[self.turn % 2].position = next;
            self.turn = self.turn + 1;
        }
        move_result
    }

    #[cfg(test)]
    fn test_version_find_all_squares_relevant_for_pawn(
        &self,
        pawn_index: usize,
    ) -> RelevantSquares {
        let distance_to_finish = self.distance_to_finish_line(pawn_index);
        let (allowed_walls, effects) =
            self.allowed_walls_for_pawn(self.pawns[pawn_index], &distance_to_finish);
        RelevantSquares::new(self, pawn_index, &distance_to_finish, &allowed_walls)
    }

    fn two_points_connected(
        &self,
        direction: WallDirection,
        location: Position,
        cache: &[NextMovesCache; 2],
        distances_from_pawns: [Option<i8>; 2],
    ) -> bool {
        if !self.walls.connect_on_two_points(direction, location, None) {
            return false;
        }
        let pawn_zero_pocket = distances_from_pawns[0].is_none()
            || cache[0].allowed_walls_for_pawn[(direction, location)] == WallType::Pocket;
        let pawn_one_pocket = distances_from_pawns[1].is_none()
            || cache[1].allowed_walls_for_pawn[(direction, location)] == WallType::Pocket;
        !(pawn_zero_pocket && pawn_one_pocket)
    }

    fn wall_score(
        &self,
        direction: WallDirection,
        location: Position,
        cache: &[NextMovesCache; 2],
        pawn_index: usize,
    ) -> i8 {
        let wall = (direction, location);
        let score_pawn_0 = cache[0].allowed_walls_for_pawn[wall].wall_score();
        let score_pawn_1 = cache[1].allowed_walls_for_pawn[wall].wall_score();
        let score = if score_pawn_0 == score_pawn_1 {
            score_pawn_0
        } else {
            return if pawn_index == 0 {
                score_pawn_0 - score_pawn_1
            } else {
                score_pawn_1 - score_pawn_0
            };
        };
        score / 3
    }

    fn wall_move_metrics(
        &self,
        wall_direction: WallDirection,
        location: Position,
        cache: &[NextMovesCache; 2],
        allowed_walls: &Walls,
        wall_value: i8,
    ) -> Option<WallMoveMetrics> {
        let distances_from_pawns: [Option<i8>; 2] = [
            closest_square_distance(location, &cache[0].relevant_squares.squares),
            closest_square_distance(location, &cache[1].relevant_squares.squares),
        ];
        // This wall will never be relevant for either pawn
        if distances_from_pawns[0].is_none() && distances_from_pawns[1].is_none() {
            return None;
        }
        let wall_blocks_wall = allowed_walls.is_allowed(wall_direction.orthogonal(), location)
            && self.two_points_connected(
                wall_direction.orthogonal(),
                location,
                cache,
                distances_from_pawns,
            );
        let wall_connected_two_points =
            self.two_points_connected(wall_direction, location, cache, distances_from_pawns);

        let distance_from_our_pawn = distances_from_pawns[self.turn % 2].map_or(100, |x| x);
        let distance_from_opponent_pawn =
            distances_from_pawns[(self.turn + 1) % 2].map_or(100, |x| x);

        let mut distance_from_shortest_path = [100, 100];
        for i in 0..2 {
            let shortest_path_length = cache[i].distances_to_finish.dist
                [cache[i].relevant_squares.distance_zero_square]
                .unwrap();
            // Here we calculate how long the route to the finish through this square would be
            let route_to_finish_length = closest_square_sum_distance(
                location,
                &cache[i].relevant_squares.squares,
                &cache[i].distances_to_finish.dist,
            );
            // Here we need to consider the distance walked by the pawn...
            if let Some(route_to_finish_length) = route_to_finish_length {
                distance_from_shortest_path[i] = route_to_finish_length - shortest_path_length;
            }
        }
        let number_of_walls_wall_connect = self
            .walls
            .number_of_walls_wall_connect(wall_direction, location);

        Some(WallMoveMetrics {
            distance_from_our_pawn,
            distance_from_opponent_pawn,
            shortest_path_difference: wall_value,
            wall_blocks_wall,
            wall_connected_two_points,
            number_of_walls_wall_connect,
            distance_from_our_shortest_path: distance_from_shortest_path[self.turn % 2],
            distance_from_their_shortest_path: distance_from_shortest_path[(self.turn + 1) % 2],
            wall_score: self.wall_score(wall_direction, location, cache, self.turn % 2),
        })
    }

    pub fn routes_to_finish(
        &self,
        start_position: Position,
        distances: DistancesToFinish,
        start_distance: i8,
    ) -> [[i8; 9]; 9] {
        let mut routes_to_finish = [[-3; 9]; 9];
        let mut queu = VecDeque::new();
        routes_to_finish[start_position] = start_distance;
        queu.push_back(start_position);
        while let Some(pos) = queu.pop_front() {
            for pawn_move in PAWN_MOVES_DOWN_LAST {
                if self.open_routes.is_open(pos, pawn_move) {
                    let new_pos = pos.add_move(pawn_move);
                    if distances.dist[new_pos].unwrap() >= distances.dist[pos].unwrap() {
                        continue;
                    }
                    if routes_to_finish[new_pos] == -3 {
                        queu.push_back(new_pos);
                        routes_to_finish[new_pos] = routes_to_finish[pos] + 1;
                    }
                }
            }
        }

        routes_to_finish
    }

    pub fn best_next_greedy_move(&self) -> Option<(Move, i8)> {
        let mut best_move = None;
        let mut best_score = -123;
        let distance_to_finish = self.distance_to_finish_line(self.turn % 2).dist;
        let current_distance = distance_to_finish[self.pawns[self.turn % 2].position].unwrap();
        for next_pawn_move in self.next_pawn_moves() {
            let score = current_distance - distance_to_finish[next_pawn_move.1].unwrap();
            if next_pawn_move.1.row == self.pawns[self.turn % 2].goal_row {
                // only winner
                return Some((
                    Move::PawnMove(next_pawn_move.0 .0, next_pawn_move.0 .1),
                    score,
                ));
            }
            if score > best_score {
                best_move = Some((
                    Move::PawnMove(next_pawn_move.0 .0, next_pawn_move.0 .1),
                    score,
                ));
                best_score = score;
            }
        }
        if self.pawns[self.turn % 2].number_of_walls_left == 0 {
            return best_move;
        }

        let allowed_walls = self.allowed_walls();
        let walls_on_shortest_paths = [
            self.allowed_walls_blocking_shortest_path(0, &allowed_walls),
            self.allowed_walls_blocking_shortest_path(1, &allowed_walls),
        ];
        for wall_move in &walls_on_shortest_paths[(self.turn + 1) % 2].1 {
            let wall_diff_score = self.shortest_path_distance_difference_pawn(
                *wall_move,
                self.turn % 2,
                &walls_on_shortest_paths,
            );
            if wall_diff_score > best_score {
                best_move = Some((Move::Wall(wall_move.0, wall_move.1), wall_diff_score));
                best_score = wall_diff_score;
            }
        }

        best_move
    }

    // return a list of allowed moves, if there is a winning move, only return that one.
    pub fn next_moves_with_scoring(
        &self,
        use_scoring: bool,
        small_rng: &mut SmallRng,
        cache: &[NextMovesCache; 2],
    ) -> Vec<(Move, i8)> {
        let mut next_moves = vec![];
        let distance_to_finish = &cache[self.turn % 2].distances_to_finish.dist;
        let current_distance = distance_to_finish[self.pawns[self.turn % 2].position].unwrap();
        let distance_from_pawn = &cache[self.turn % 2].relevant_squares.squares;

        let is_mirrored = self.is_mirrored();
        for next_pawn_move in self.next_pawn_moves() {
            if is_mirrored {
                if next_pawn_move.0 .0 == PawnMove::Left {
                    continue;
                }
                if let Some(second_move) = next_pawn_move.0 .1 {
                    if second_move == PawnMove::Left {
                        continue;
                    }
                }
            }
            let mut score = current_distance - distance_to_finish[next_pawn_move.1].unwrap();
            if next_pawn_move.1.row == self.pawns[self.turn % 2].goal_row {
                // only winner, this gives some weird results at the end though...
                return vec![(
                    Move::PawnMove(next_pawn_move.0 .0, next_pawn_move.0 .1),
                    score,
                )];
            }
            // Is a spot we logically don't want to go. // We might need to include this if there are no moves possible?
            // In case the opponent has no walls left, we shouldn't do this, cause then we se every distance from pawn at None. Cause no squares are relevant anymore
            if self.pawns[(self.turn + 1) % 2].number_of_walls_left != 0 {
                if distance_from_pawn[next_pawn_move.1].is_none() {
                    score = -100;
                }
            }
            next_moves.push((
                Move::PawnMove(next_pawn_move.0 .0, next_pawn_move.0 .1),
                score,
            ));
        }
        if next_moves.iter().any(|x| x.1 >= 0) {
            // Now we want to remove everything that does have -100 as a score.
            next_moves.retain(|x| x.1 >= 0);
        } else {
            for next_move in next_moves.iter_mut() {
                next_move.1 = 0;
            }
        }

        // In case the opponent has no walls left, walk shortest path.
        if self.pawns[(self.turn + 1) % 2].number_of_walls_left == 0 {
            // Might be multiple shortest paths, then we want to be able to choose
            next_moves.sort_by_key(|x| -x.1);
            let best_score = next_moves[0].1;
            next_moves.retain(|x| x.1 == best_score);
        }
        // In case no walls left, we can only do pawn moves.
        if self.pawns[self.turn % 2].number_of_walls_left == 0 {
            for next_move in next_moves.iter_mut() {
                next_move.1 += 3;
                next_move.1 = next_move.1.max(0);
            }
            return next_moves;
        }

        let max_col = if is_mirrored { 4 } else { 8 };
        // in case opponent has no walls left, we will only place walls directly next to them.
        // TODO look into excluding more...
        // If we are close to opponent, consider walls, cause in that case in might affect a jump.
        // So if we have move that puts us next to opponent, our one of our moves is a jump move
        if self.pawns[(self.turn + 1) % 2].number_of_walls_left == 0 {
            let mut close_opponent = false;
            for game_move in &next_moves {
                match game_move.0 {
                    Move::PawnMove(first, second) => {
                        if let Some(_) = second {
                            close_opponent = true;
                            break;
                        }
                        let next_pos = self.pawns[self.turn % 2].position.add_move(first);
                        if let Some(pawn_move) = self.pawns[(self.turn + 1) % 2]
                            .position
                            .substract_pos(&next_pos)
                        {
                            if self.open_routes.is_open(next_pos, pawn_move) {
                                close_opponent = true;
                            }
                            break;
                        }
                    }

                    _ => (),
                };
            }
            if !close_opponent {
                for row in 0..8 {
                    for col in 0..max_col {
                        let location = Position {
                            row: row as i8,
                            col: col as i8,
                        };
                        let pos = Position { row, col };
                        for direction in [WallDirection::Horizontal, WallDirection::Vertical] {
                            let square_distance = closest_square_distance(
                                pos,
                                &cache[(self.turn + 1) % 2].relevant_squares.squares,
                            );
                            if let Some(sq_dist) = square_distance {
                                let sq_dist = sq_dist
                                    + cache[(self.turn + 1) % 2]
                                        .relevant_squares
                                        .dist_walked_unhindered;
                                if sq_dist <= 1
                                    && cache[0].allowed_walls_for_pawn[(direction, location)]
                                        .is_allowed()
                                    && cache[1].allowed_walls_for_pawn[(direction, location)]
                                        .is_allowed()
                                {
                                    {
                                        next_moves
                                            .push((Move::Wall(direction, location), -sq_dist));
                                    }
                                }
                            }
                        }
                    }
                }
                next_moves.sort_by_key(|x| -x.1);
                return next_moves;
            }
        }

        let allowed_walls = allowed_walls(cache);
        // TODO: use cache for this calculation as well.
        let walls_on_shortest_paths = [
            self.allowed_walls_blocking_shortest_path(0, &allowed_walls),
            self.allowed_walls_blocking_shortest_path(1, &allowed_walls),
        ];

        for row in 0..8 {
            for col in 0..max_col {
                let location = Position {
                    row: row as i8,
                    col: col as i8,
                };
                for direction in [WallDirection::Horizontal, WallDirection::Vertical].iter() {
                    if allowed_walls.is_allowed(*direction, location) {
                        let wall_diff_score = self.shortest_path_distance_difference_pawn(
                            (*direction, location),
                            self.turn % 2,
                            &walls_on_shortest_paths,
                        );
                        if let Some(wall_metrics) = self.wall_move_metrics(
                            *direction,
                            location,
                            cache,
                            &allowed_walls,
                            wall_diff_score,
                        ) {
                            if wall_metrics.is_probable() {
                                next_moves.push((
                                    Move::Wall(*direction, location),
                                    wall_metrics.move_score(),
                                ));
                            } else {
                                next_moves.push((Move::Wall(*direction, location), -100));
                            }
                        }
                    }
                }
            }
        }
        if !use_scoring {
            for next_move in next_moves.iter_mut() {
                next_move.1 = 3;
            }
        } else {
            // We sort high value moves to the front
            next_moves.sort_by_key(|x| -x.1);
            //// only take the first 20 options.
            //let to_take = std::cmp::min(20, next_moves.len());
            //next_moves = next_moves[..to_take].to_vec();
            let mut start = 0;
            while start < next_moves.len() {
                let mut end = start + 1;
                while end < next_moves.len() && next_moves[end].1 == next_moves[start].1 {
                    end += 1;
                }
                // Shuffle this block
                next_moves[start..end].shuffle(small_rng);
                start = end; // Move to the next block
            }
        }
        next_moves
    }

    pub fn previous_boards(&self) -> Vec<(Board, Move)> {
        let mut walls = vec![];
        for row in 0..8 {
            for col in 0..8 {
                let pos = Position { row, col };
                if self.walls.horizontal[pos] {
                    walls.push((WallDirection::Horizontal, pos));
                }
                if self.walls.vertical[pos] {
                    walls.push((WallDirection::Vertical, pos));
                }
            }
        }
        let mut prev_boards = vec![];
        for wall in walls {
            let mut prev_board = self.clone();
            prev_board.turn -= 1;
            prev_board.walls.wall_not_allowed(wall.0, wall.1);
            prev_board.pawns[prev_board.turn % 2].number_of_walls_left += 1;
            prev_boards.push((
                Board::decode(&prev_board.encode()).unwrap(),
                Move::Wall(wall.0, wall.1),
            ));
        }

        let mut prev_board = self.clone();
        prev_board.turn -= 1;
        for pawn_move in prev_board.next_pawn_moves() {
            let mut prev_board = prev_board.clone();
            prev_board.move_pawn(pawn_move.0);
            prev_board.turn -= 1;
            prev_boards.push((
                Board::decode(&prev_board.encode()).unwrap(),
                Move::PawnMove(
                    pawn_move.0 .0.opposite_move(),
                    pawn_move.0 .1.map(|x| x.opposite_move()),
                ),
            ));
        }

        prev_boards
    }

    pub fn game_move_possible(&self, game_move: Move) -> MoveResult {
        match game_move {
            Move::PawnMove(pawn_move, second_pawn_move) => {
                self.can_move_pawn((pawn_move, second_pawn_move)).0
            }
            Move::Wall(direction, location) => {
                if self.can_place_wall(direction, location) {
                    MoveResult::Valid
                } else {
                    MoveResult::Invalid
                }
            }
        }
    }

    pub fn minimal_board(&self) -> MinimalBoard {
        MinimalBoard {
            turn: self.turn,
            pawns: [self.pawns[0].position, self.pawns[1].position],
            walls: self.walls.clone(),
        }
    }

    pub fn copy_game_move(&self, game_move: Move) -> (MoveResult, Option<MinimalBoard>) {
        let move_result = self.game_move_possible(game_move);
        if move_result == MoveResult::Invalid {
            return (move_result, None);
        }
        let mut minimal_board = self.minimal_board();
        minimal_board.game_move(game_move);
        (move_result, Some(minimal_board))
    }

    pub fn copy_game_move_unchecked(&self, game_move: Move) -> MinimalBoard {
        let mut minimal_board = self.minimal_board();
        minimal_board.game_move(game_move);
        minimal_board
    }

    pub fn game_move(&mut self, game_move: Move) -> MoveResult {
        match game_move {
            Move::PawnMove(pawn_move, second_pawn_move) => {
                self.move_pawn((pawn_move, second_pawn_move))
            }
            Move::Wall(direction, location) => {
                if self.place_wall(direction, location) {
                    MoveResult::Valid
                } else {
                    MoveResult::Invalid
                }
            }
        }
    }

    pub fn is_possible_next_pawn_location(
        &self,
        row: usize,
        col: usize,
    ) -> Option<(PawnMove, Option<PawnMove>)> {
        self.next_pawn_moves()
            .into_iter()
            .find(|(_pawn_move, position)| {
                position
                    == &Position {
                        row: row as i8,
                        col: col as i8,
                    }
            })
            .map(|x| x.0)
    }

    pub fn can_pawn_win(&self) -> bool {
        let pawn = self.pawns[self.turn % 2];
        if (pawn.position.row - pawn.goal_row).abs() > 2 {
            return false;
        }
        self.next_pawn_moves()
            .into_iter()
            .any(|(_pawn_move, position)| position.row == pawn.goal_row)
    }

    pub fn next_pawn_moves(&self) -> Vec<((PawnMove, Option<PawnMove>), Position)> {
        let pawn_index = self.turn % 2;
        let pawn = self.pawns[pawn_index];
        let mut positions = Vec::new();
        for pawn_move in [
            PawnMove::Up,
            PawnMove::Down,
            PawnMove::Left,
            PawnMove::Right,
        ] {
            if self.open_routes.is_open(pawn.position, pawn_move) {
                // if the next position is occupied by the other pawn, we can jump over it. If there is a wall behind the other pawn, we can jump diagonally.
                let next_position = pawn.position.add_move(pawn_move);
                if self
                    .is_pawn(next_position.row as usize, next_position.col as usize)
                    .is_some()
                {
                    if self.open_routes.is_open(next_position, pawn_move) {
                        positions.push((
                            (pawn_move, Some(pawn_move)),
                            next_position.add_move(pawn_move),
                        ));
                    } else {
                        for second_pawn_move in pawn_move.orthogonal_moves() {
                            if self.open_routes.is_open(next_position, second_pawn_move) {
                                positions.push((
                                    (pawn_move, Some(second_pawn_move)),
                                    next_position.add_move(second_pawn_move),
                                ));
                            }
                        }
                    }
                } else {
                    positions.push(((pawn_move, None), pawn.position.add_move(pawn_move)));
                }
            }
        }
        positions
    }

    // Check if the board is vertically mirrored
    // To do this we check if all the walls have a wall on the mirrored side and whether both pawn have col = 4
    fn is_mirrored(&self) -> bool {
        for row in 0..8 {
            for col in 0..8 {
                let location = Position {
                    row: row as i8,
                    col: col as i8,
                };
                if self.walls.horizontal[location] {
                    if !self.walls.horizontal[row][8 - col - 1] {
                        return false;
                    }
                }
                if self.walls.vertical[location] {
                    if !self.walls.vertical[row][8 - col - 1] {
                        return false;
                    }
                }
            }
        }
        self.pawns[0].position.col == 4 && self.pawns[1].position.col == 4
    }

    // return a mirrored version of the board
    pub fn encode_mirror(&self) -> String {
        let mut new_board = self.clone();
        new_board.pawns[0].position.col = 8 - new_board.pawns[0].position.col;
        new_board.pawns[1].position.col = 8 - new_board.pawns[1].position.col;
        new_board.walls = self.walls.mirror();
        new_board.encode()
    }

    // returns all legal next moves. Except the the ones that are mirrored from each other.
    pub fn next_non_mirrored_moves(&self) -> Vec<Move> {
        let mut next_moves = vec![];
        let is_mirrored = self.is_mirrored();
        for pawn_move in self.next_pawn_moves() {
            if is_mirrored {
                if pawn_move.0 .0 == PawnMove::Left {
                    continue;
                }
                if let Some(second_move) = pawn_move.0 .1 {
                    if second_move == PawnMove::Left {
                        continue;
                    }
                }
            }
            next_moves.push(Move::PawnMove(pawn_move.0 .0, pawn_move.0 .1));
        }
        let number_columns = if is_mirrored { 4 } else { 8 };
        if self.pawns[self.turn % 2].number_of_walls_left > 0 {
            for row in 0..8 {
                for col in 0..number_columns {
                    let location = Position {
                        row: row as i8,
                        col: col as i8,
                    };
                    for direction in [WallDirection::Horizontal, WallDirection::Vertical].iter() {
                        if self.can_place_wall(*direction, location) {
                            next_moves.push(Move::Wall(*direction, location));
                        }
                    }
                }
            }
        }

        next_moves
    }

    pub fn move_along_shortest_path_considering_distances(
        &self,
        pawn_index: usize,
        distances: &[[Option<i8>; 9]; 9],
    ) -> Option<(PawnMove, Option<PawnMove>)> {
        let mut minimum_move: Option<((PawnMove, Option<PawnMove>), i8)> = None;
        for (pawn_move, position) in self.next_pawn_moves() {
            if let Some(distance) = distances[position] {
                if minimum_move.is_none() || distance < minimum_move.unwrap().1 {
                    minimum_move = Some((pawn_move, distance));
                }
            }
            if position == self.pawns[pawn_index].position {
                return Some(pawn_move);
            }
        }
        minimum_move.map(|x| x.0)
    }

    pub fn move_along_shortest_path(
        &self,
        pawn_index: usize,
    ) -> Option<(PawnMove, Option<PawnMove>)> {
        let distances = self.distance_to_finish_line(pawn_index);
        self.move_along_shortest_path_considering_distances(pawn_index, &distances.dist)
    }

    pub fn shortest_path(&self, pawn_index: usize) -> Vec<(PawnMove, Position)> {
        let distances = self.distance_to_finish_line(pawn_index);
        // Now we will walk from the pawn location to the finish line, following the neighboring tiles where the distance decreases most.
        let mut current = self.pawns[pawn_index].position;
        let mut shortest_path = vec![];
        while current.row != self.pawns[pawn_index].goal_row {
            let mut minimum_distance = None;
            let mut minimum_move = None;
            for pawn_move in [
                PawnMove::Up,
                PawnMove::Down,
                PawnMove::Left,
                PawnMove::Right,
            ] {
                let next = current.add_move(pawn_move);
                if self.open_routes.is_open(current, pawn_move) {
                    if let Some(distance) = distances.dist[next] {
                        if minimum_distance.is_none() || distance < minimum_distance.unwrap() {
                            minimum_distance = Some(distance);
                            minimum_move = Some(pawn_move);
                        }
                    }
                }
            }
            if let Some(minimum_move) = minimum_move {
                shortest_path.push((minimum_move, current));
                current = current.add_move(minimum_move);
            } else {
                break;
            }
        }
        shortest_path
    }

    pub fn allowed_walls_blocking_only_opponents_path(
        &self,
        pawn_index: usize,
        allowed_walls: &Walls,
    ) -> Vec<(WallDirection, Position)> {
        let walls_blocking_our_path = self
            .allowed_walls_blocking_shortest_path(pawn_index, allowed_walls)
            .1;

        let mut walls_blocking_opponents_path = self
            .allowed_walls_blocking_shortest_path((pawn_index + 1) % 2, allowed_walls)
            .1;

        walls_blocking_opponents_path.retain(|x| !walls_blocking_our_path.contains(x));
        walls_blocking_opponents_path
    }

    // Return the path length and all the walls blocking it.
    pub fn allowed_walls_blocking_shortest_path(
        &self,
        pawn_index: usize,
        allowed_walls: &Walls,
    ) -> (i8, Vec<(WallDirection, Position)>) {
        let shortest_path = self
            .open_routes
            .shortest_path_from_dest_row_to_pawn(self.pawns[pawn_index]);
        let path_len = shortest_path.len() as i8;
        let mut walls = vec![];
        for (pawn_move, location) in shortest_path {
            let Position { row, col } = location;
            let row = row as usize;
            let col = col as usize;
            match pawn_move {
                PawnMove::Up => {
                    // we can not check this, cause pawn move up
                    if col < 8 && self.allowed_walls.horizontal[row][col] {
                        walls.push((WallDirection::Horizontal, location));
                    }
                    if col >= 1 && self.allowed_walls.horizontal[row][col - 1] {
                        walls.push((WallDirection::Horizontal, (row, col - 1).into()));
                    }
                }
                PawnMove::Down => {
                    if col < 8 && self.allowed_walls.horizontal[row - 1][col] {
                        walls.push((WallDirection::Horizontal, (row - 1, col).into()));
                    }
                    if col >= 1 && self.allowed_walls.horizontal[row - 1][col - 1] {
                        walls.push((WallDirection::Horizontal, (row - 1, col - 1).into()));
                    }
                }
                PawnMove::Left => {
                    if row < 8 && self.allowed_walls.vertical[row][col - 1] {
                        walls.push((WallDirection::Vertical, (row, col - 1).into()));
                    }
                    if row >= 1 && self.allowed_walls.vertical[row - 1][col - 1] {
                        walls.push((WallDirection::Vertical, (row - 1, col - 1).into()));
                    }
                }
                PawnMove::Right => {
                    if row < 8 && self.allowed_walls.vertical[row][col] {
                        walls.push((WallDirection::Vertical, (row, col).into()));
                    }
                    if row >= 1 && self.allowed_walls.vertical[row - 1][col] {
                        walls.push((WallDirection::Vertical, (row - 1, col).into()));
                    }
                }
            }
        }
        walls.retain(|x| allowed_walls[*x]);
        (path_len, walls)
    }

    pub fn distance_to_finish_line(&self, pawn_index: usize) -> DistancesToFinish {
        let pawn = self.pawns[pawn_index];
        let mut visited: [[bool; 9]; 9] = [[false; 9]; 9];
        let mut distance: [[Option<i8>; 9]; 9] = [[None; 9]; 9];
        let mut queue: ArrayDeque<Position, 40> = ArrayDeque::new();
        //let mut queue: VecDeque<Position> = VecDeque::new();
        for col in 0..9 {
            queue
                .push_back(Position {
                    row: pawn.goal_row,
                    col,
                })
                .unwrap();
            distance[pawn.goal_row as usize][col as usize] = Some(0);
            visited[pawn.goal_row as usize][col as usize] = true;
        }
        while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            let current_distance = distance[current.row as usize][current.col as usize];
            for pawn_move in [
                PawnMove::Up,
                PawnMove::Down,
                PawnMove::Left,
                PawnMove::Right,
            ] {
                if self.open_routes.is_open(current, pawn_move) {
                    let next = current.add_move(pawn_move);

                    if !visited[next] {
                        visited[next] = true;
                        let next_distance = current_distance.map(|x| x + 1);
                        distance[next] = next_distance;
                        queue.push_back(next).unwrap();
                    }
                }
            }
        }
        DistancesToFinish { dist: distance }
    }

    fn is_wall_allowed_pawn_new(
        &self,
        wall: (WallDirection, Position),
        pawn: Pawn,
        distances_to_finish: &DistancesToFinish,
    ) -> (WallType, Option<WallEffect>) {
        if !self.allowed_walls[wall] {
            return (WallType::Impossible, None);
        }

        let (left, middle_top, middle_bottom, right) =
            self.walls.connected_points(wall.0, wall.1, None);
        if (left as u8 + (middle_top || middle_bottom) as u8 + right as u8) <= 1 {
            return (WallType::Allowed(0), None);
        }
        if let Some(wall_sides) = self.walls.sides_of_wall(wall.0, wall.1) {
            let vertical_last_row = if pawn.goal_row == 0 {
                wall.1.row == 0 && wall.0 == WallDirection::Vertical
            } else {
                wall.1.row == 7 && wall.0 == WallDirection::Vertical
            };
            let mut open_routes = self.open_routes.clone();
            open_routes.update_open(wall.0, wall.1);
            let pocket_result_0 = open_routes.check_if_pocket(wall_sides[0], pawn, wall_sides[1]);
            if pocket_result_0.reached_otherside.is_some() {
                return (
                    WallType::Allowed(0),
                    Some(WallEffect::DistanceOtherside(
                        pocket_result_0.reached_otherside.unwrap(),
                    )),
                );
            }
            if pocket_result_0.is_pocket {
                return (WallType::Pocket, None);
            }
            if !pocket_result_0.wall_allowed {
                return (WallType::Unallowed, None);
            }
            let pocket_result_1 = open_routes.check_if_pocket(wall_sides[1], pawn, wall_sides[0]);
            if pocket_result_1.is_pocket {
                return (WallType::Pocket, None);
            }
            let area_left = if pocket_result_0.pawn_seen {
                pocket_result_0.area_seen
            } else {
                pocket_result_1.area_seen
            };
            if pocket_result_0.wall_allowed && pocket_result_1.wall_allowed {
                if vertical_last_row {
                    return (WallType::Pocket, None);
                } else {
                    return (
                        WallType::Allowed(0),
                        Some(WallEffect::AreaLeftPawn(area_left)),
                    );
                }
            } else {
                (WallType::Unallowed, None)
            }
        } else {
            if distances_to_finish.wall_parrallel(wall.0, wall.1) {
                // TODO:
                return (WallType::Allowed(0), None);
            }
            let wall_distance_to_finish =
                closest_square_distance(wall.1, &distances_to_finish.dist);
            let pawn_distance_to_finish = distances_to_finish.dist[pawn.position];
            // In this case this wall can never block this pawn.
            if wall_distance_to_finish.is_none()
                || wall_distance_to_finish >= pawn_distance_to_finish
            {
                return (WallType::Allowed(0), None);
            }
            let mut open_routes = self.open_routes.clone();
            open_routes.update_open(wall.0, wall.1);
            if open_routes
                .find_path_for_pawn_to_dest_row(pawn, false)
                .is_none()
            {
                return (WallType::Unallowed, None);
            }
            return (WallType::Allowed(0), None);
        }
    }

    /// Here we want to check whether a wall is allowed considering whether it blocks the path of the pawn to the other side.
    fn is_wall_allowed(&self, direction: WallDirection, location: Position) -> bool {
        // here we check the allowed falls, if the wall is not allowed, we return false.
        if !self.allowed_walls[(direction, location)] {
            return false;
        }

        if !self.walls.connect_on_two_points(direction, location, None) {
            return true;
        }
        let mut open_routes = self.open_routes.clone();
        open_routes.update_open(direction, location);
        for i in 0..2 {
            if open_routes
                .find_path_for_pawn_to_dest_row(self.pawns[i], false)
                .is_none()
            {
                return false;
            }
        }
        return true;
    }
}

#[derive(Debug)]
pub struct WallMoveMetrics {
    distance_from_our_pawn: i8,
    distance_from_opponent_pawn: i8,
    shortest_path_difference: i8,
    wall_blocks_wall: bool,
    wall_connected_two_points: bool,
    number_of_walls_wall_connect: i8,
    distance_from_our_shortest_path: i8,
    distance_from_their_shortest_path: i8,
    wall_score: i8,
}

impl WallMoveMetrics {
    pub fn print_csv_column_names() {
        println!(
            "{},{},{},{}, {}, {}, {}, {}, {}",
            "distance_from_our_pawn",
            "distance_from_opponent_pawn",
            "shortest_path_difference",
            "wall_blocks_wall",
            "wall_connected_two_points",
            "number_of_walls_wall_connect",
            "distance_from_shortest_path",
            "win_rate",
            "number_of_visits"
        );
    }

    pub fn print_as_csv_line(&self, scores: Option<(f64, usize)>) {
        let win_rate = scores.map(|x| x.0 / x.1 as f64);
        let win_rate = win_rate.map(|x| x.to_string()).unwrap_or("".to_string());
        println!(
            "{},{},{},{}, {}, {}, {}, {}, {}, {}",
            self.distance_from_our_pawn,
            self.distance_from_opponent_pawn,
            self.shortest_path_difference,
            self.wall_blocks_wall,
            self.wall_connected_two_points,
            self.number_of_walls_wall_connect,
            self.distance_from_our_shortest_path,
            self.distance_from_their_shortest_path,
            win_rate,
            scores.map(|x| x.1).unwrap_or(0),
        );
    }
    pub fn is_probable(&self) -> bool {
        if self.distance_from_our_pawn <= 1
            || self.distance_from_opponent_pawn <= 1 && self.distance_from_their_shortest_path <= 1
        {
            return true;
        }
        // THe wall either potentially blocks a path, or it stops a wall from blocking of a path.
        if self.wall_connected_two_points || self.wall_blocks_wall {
            return true;
        }
        if self.shortest_path_difference >= 2 {
            return true;
        }
        if self.shortest_path_difference >= 1
            && (self.distance_from_opponent_pawn <= 3 || self.distance_from_our_pawn <= 3)
        {
            return true;
        }
        if self.distance_from_our_shortest_path <= 1 || self.distance_from_their_shortest_path <= 1
        {
            self.number_of_walls_wall_connect >= 1
        } else {
            false
        }
    }

    pub fn move_score(&self) -> i8 {
        self.shortest_path_difference + self.wall_score
    }
}

// Here we gonna define some metrics for wall moves, so to use later in the AI.
impl Board {
    pub fn move_metrics(
        &self,
        game_move: Move,
        cache: &[NextMovesCache; 2],
        allowed_walls: &Walls,
        wall_value: i8,
    ) -> Option<WallMoveMetrics> {
        match game_move {
            Move::PawnMove(_, _) => None,
            Move::Wall(direction, location) => {
                self.wall_move_metrics(direction, location, cache, allowed_walls, wall_value)
            }
        }
    }

    #[cfg(test)]
    fn on_shortest_path_pawn(
        &self,
        wall_move: (WallDirection, Position),
        pawn_index: usize,
    ) -> bool {
        let allowed_walls = self.allowed_walls();
        let walls_on_shortest_path = self
            .allowed_walls_blocking_shortest_path(pawn_index, &allowed_walls)
            .1;
        walls_on_shortest_path.contains(&wall_move)
    }

    // TODO: Can get some massive performance improvements here, by caching the shortest path distances. And using parrallel tricks etc.
    fn shortest_path_distance_increase_pawn(
        &self,
        wall_move: (WallDirection, Position),
        pawn_index: usize,
        walls_on_shortest_path: &(i8, Vec<(WallDirection, Position)>),
    ) -> i8 {
        let mut open_routes = self.open_routes.clone();
        open_routes.update_open(wall_move.0, wall_move.1);
        let shortest_path =
            open_routes.find_path_for_pawn_to_dest_row(self.pawns[pawn_index], true);

        if shortest_path.is_none() {
            println!("This should not happen");
            println!("pawns: {:?}", self.pawns);
            println!("wall move: {:?}", wall_move);
            self.walls.pretty_print_wall();
        }
        shortest_path.unwrap() as i8 - walls_on_shortest_path.0 as i8
    }

    fn shortest_path_distance_difference_pawn(
        &self,
        wall_move: (WallDirection, Position),
        pawn_index: usize,
        walls_on_shortest_path: &[(i8, Vec<(WallDirection, Position)>); 2],
    ) -> i8 {
        let pawn_index_opp = (pawn_index + 1) % 2;
        let our_difference = if walls_on_shortest_path[pawn_index].1.contains(&wall_move) {
            self.shortest_path_distance_increase_pawn(
                wall_move,
                pawn_index,
                &walls_on_shortest_path[pawn_index],
            )
        } else {
            0
        };
        let their_difference = if walls_on_shortest_path[pawn_index_opp]
            .1
            .contains(&wall_move)
        {
            self.shortest_path_distance_increase_pawn(
                wall_move,
                pawn_index_opp,
                &walls_on_shortest_path[pawn_index_opp],
            )
        } else {
            0
        };
        their_difference - our_difference
    }
}

// Here we will create an AI, that will always walk along the shortest Path.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Move {
    PawnMove(PawnMove, Option<PawnMove>),
    Wall(WallDirection, Position),
}

#[derive(Debug)]
pub enum MirrorMoveType {
    Left,
    Right,
    Neutral,
}

impl Move {
    pub fn mirror_move_type(&self) -> MirrorMoveType {
        match self {
            Self::PawnMove(first_step, _second_step) => match first_step {
                PawnMove::Left => MirrorMoveType::Left,
                PawnMove::Right => MirrorMoveType::Right,
                _ => MirrorMoveType::Neutral,
            },
            Self::Wall(_direction, location) => {
                if location.col < 4 {
                    MirrorMoveType::Left
                } else {
                    MirrorMoveType::Right
                }
            }
        }
    }

    pub fn mirror_move(&self) -> Self {
        match self {
            Self::PawnMove(first_step, second_step) => {
                let new_first_step = match first_step {
                    PawnMove::Left => PawnMove::Right,
                    PawnMove::Right => PawnMove::Left,
                    _ => *first_step,
                };
                let new_second_step = match second_step {
                    Some(PawnMove::Left) => Some(PawnMove::Right),
                    Some(PawnMove::Right) => Some(PawnMove::Left),
                    _ => *second_step,
                };
                Self::PawnMove(new_first_step, new_second_step)
            }
            Self::Wall(direction, location) => {
                let new_col = 7 - location.col;
                Self::Wall(
                    *direction,
                    Position {
                        row: location.row,
                        col: new_col,
                    },
                )
            }
        }
    }
}

#[cfg(test)]
mod test {
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::Instant;
    #[cfg(target_arch = "wasm32")]
    use web_time::Instant;

    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_wall_on_path() {
        let mut board = Board::new();

        board.pawns[0].position = Position { row: 0, col: 4 };
        board.pawns[0].goal_row = 8;

        assert!(board.on_shortest_path_pawn((WallDirection::Horizontal, (7, 3).into()), 0));
        assert!(board.on_shortest_path_pawn((WallDirection::Horizontal, (7, 4).into()), 0));

        assert!(!board.on_shortest_path_pawn((WallDirection::Vertical, (7, 3).into()), 0));
        assert!(!board.on_shortest_path_pawn((WallDirection::Vertical, (7, 4).into()), 0));

        assert!(!board.on_shortest_path_pawn((WallDirection::Horizontal, (7, 0).into()), 0));
        assert!(!board.on_shortest_path_pawn((WallDirection::Horizontal, (7, 1).into()), 0));
        assert!(!board.on_shortest_path_pawn((WallDirection::Horizontal, (7, 2).into()), 0));

        board.place_wall(WallDirection::Horizontal, (7, 3).into());

        assert!(board.on_shortest_path_pawn((WallDirection::Horizontal, (7, 5).into()), 0));
    }

    #[test]
    fn test_wall_allowed() {
        let mut board = Board::new();

        // in this array we store the wall moves we are going to make
        let walls = [
            (WallDirection::Horizontal, Position { row: 5, col: 0 }),
            (WallDirection::Horizontal, Position { row: 5, col: 4 }),
            (WallDirection::Horizontal, Position { row: 5, col: 6 }),
            (WallDirection::Horizontal, Position { row: 5, col: 2 }),
        ];

        let mut results = vec![];
        for (direction, location) in walls {
            let start = Instant::now();
            let allowed = board.place_wall(direction, location);
            results.push(allowed);
            println!(
                "is {:?} wall on position {:?} allowed: {}, took: {:?}",
                direction,
                location,
                allowed,
                start.elapsed()
            );
        }
        assert_eq!(results, vec![true, true, true, true]);
    }

    #[test]
    fn test_wall_not_allowed() {
        let mut board = Board::new();

        // in this array we store the wall moves we are going to make
        let walls = [
            (WallDirection::Horizontal, Position { row: 5, col: 0 }),
            (WallDirection::Horizontal, Position { row: 5, col: 2 }),
            (WallDirection::Horizontal, Position { row: 5, col: 4 }),
            (WallDirection::Horizontal, Position { row: 5, col: 6 }),
            (WallDirection::Vertical, Position { row: 5, col: 7 }),
            (WallDirection::Horizontal, Position { row: 4, col: 7 }),
            (WallDirection::Horizontal, Position { row: 6, col: 7 }),
            (WallDirection::Horizontal, Position { row: 1, col: 7 }),
        ];

        let mut results = vec![];
        for (direction, location) in walls {
            let start = Instant::now();
            let allowed = board.place_wall(direction, location);
            results.push(allowed);
            println!(
                "is {:?} wall on position {:?} allowed: {}, took: {:?}",
                direction,
                location,
                allowed,
                start.elapsed()
            );
        }
        assert_eq!(
            results,
            vec![true, true, true, true, true, false, false, true]
        );
    }

    #[test]
    fn test_next_moves() {
        let mut board = Board::new();

        // in this array we store the wall moves we are going to make
        let walls = [
            (WallDirection::Horizontal, Position { row: 5, col: 0 }),
            (WallDirection::Horizontal, Position { row: 5, col: 2 }),
            (WallDirection::Horizontal, Position { row: 5, col: 4 }),
            (WallDirection::Horizontal, Position { row: 5, col: 6 }),
            (WallDirection::Vertical, Position { row: 5, col: 7 }),
        ];
        for (direction, location) in walls {
            let _allowed = board.place_wall(direction, location);
        }
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        let start = Instant::now();
        let next_moves = board.next_moves_with_scoring(true, &mut SmallRng::from_entropy(), &cache);
        println!(
            "next moves number: {:?}, took: {:?}",
            next_moves.len(),
            start.elapsed()
        );
    }
    #[test]
    fn test_wrong_moves() {
        let mut board = Board::new();

        //        [None, Some(Wall(Vertical, (7, 4))), Some(Wall(Horizontal, (6, 3))), Some(Wall(Vertical, (7, 3)))]

        // in this array we store the wall moves we are going to make
        let walls = [
            (WallDirection::Vertical, Position { row: 7, col: 4 }),
            (WallDirection::Horizontal, Position { row: 6, col: 3 }),
            (WallDirection::Vertical, Position { row: 7, col: 3 }),
        ];
        let mut results = vec![];
        for (direction, location) in walls {
            let start = Instant::now();
            let allowed = board.place_wall(direction, location);
            results.push(allowed);
            println!(
                "is {:?} wall on position {:?} allowed: {}, took: {:?}",
                direction,
                location,
                allowed,
                start.elapsed()
            );
        }
        assert_eq!(vec![true, true, false], results);
    }

    #[test]
    fn test_walls_blocking_opponent() {
        let mut board = Board::new();

        let start = Instant::now();
        let walls_on_path = board
            .open_routes
            .shortest_path_from_dest_row_to_pawn(board.pawns[0]);
        println!("walls on path took: {:?}", start.elapsed());
        println!("walls on path {:?}", walls_on_path);

        let allowed_walls = board.allowed_walls();
        let start = Instant::now();
        let walls_blocking_opponent =
            board.allowed_walls_blocking_only_opponents_path(0, &allowed_walls);
        println!("walls blocking opponent took: {:?}", start.elapsed());
        assert_eq!(walls_blocking_opponent, vec![]);
        board.game_move(Move::PawnMove(PawnMove::Down, None));
        board.game_move(Move::PawnMove(PawnMove::Up, None));
        board.game_move(Move::PawnMove(PawnMove::Down, None));
        board.game_move(Move::PawnMove(PawnMove::Up, None));

        let start = Instant::now();
        let mut walls_blocking_opponent =
            board.allowed_walls_blocking_only_opponents_path(0, &allowed_walls);
        println!("walls blocking opponent took: {:?}", start.elapsed());
        walls_blocking_opponent.sort();
        let mut correct = vec![
            (WallDirection::Horizontal, Position { row: 1, col: 4 }),
            (WallDirection::Horizontal, Position { row: 1, col: 3 }),
            (WallDirection::Horizontal, Position { row: 0, col: 4 }),
            (WallDirection::Horizontal, Position { row: 0, col: 3 }),
        ];
        correct.sort();
        assert_eq!(walls_blocking_opponent, correct);

        //        [None, Some(Wall(Vertical, (7, 4))), Some(Wall(Horizontal, (6, 3))), Some(Wall(Vertical, (7, 3)))]

        // in this array we store the wall moves we are going to make
    }

    #[test]
    fn test_relevant_squares_blocked_in() {
        let mut board = Board::new();

        //        [None, Some(Wall(Vertical, (7, 4))), Some(Wall(Horizontal, (6, 3))), Some(Wall(Vertical, (7, 3)))]

        // in this array we store the wall moves we are going to make
        let walls = [
            (WallDirection::Vertical, Position { row: 6, col: 4 }),
            (WallDirection::Vertical, Position { row: 4, col: 4 }),
            (WallDirection::Horizontal, Position { row: 3, col: 5 }),
            (WallDirection::Horizontal, Position { row: 3, col: 7 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        let mut pawn = board.pawns[0];
        pawn.goal_row = 8;
        pawn.position = Position { row: 5, col: 5 };
        let distances_to_finish = board.distance_to_finish_line(0);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    pawn,
                    &board.allowed_walls_for_pawn(pawn, &distances_to_finish).0,
                    0
                )
                .0,
            16
        );
    }

    #[test]
    fn test_relevant_line_back() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Horizontal, Position { row: 6, col: 6 }),
            (WallDirection::Horizontal, Position { row: 6, col: 4 }),
            (WallDirection::Horizontal, Position { row: 7, col: 6 }),
            (WallDirection::Horizontal, Position { row: 7, col: 4 }),
            (WallDirection::Vertical, Position { row: 6, col: 3 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        let mut pawn = board.pawns[0];
        pawn.goal_row = 8;
        pawn.position = Position { row: 7, col: 5 };

        let distances_to_finish = board.distance_to_finish_line(0);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    pawn,
                    &board.allowed_walls_for_pawn(pawn, &distances_to_finish).0,
                    0
                )
                .0,
            5
        );

        // Now we want to test the same thing other way round
        let mut board = Board::new();

        let walls = [
            (WallDirection::Horizontal, Position { row: 1, col: 6 }),
            (WallDirection::Horizontal, Position { row: 1, col: 4 }),
            (WallDirection::Horizontal, Position { row: 0, col: 6 }),
            (WallDirection::Horizontal, Position { row: 0, col: 4 }),
            (WallDirection::Vertical, Position { row: 1, col: 3 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        let mut pawn = board.pawns[1];
        pawn.goal_row = 0;
        pawn.position = Position { row: 1, col: 5 };
        let distances_to_finish = board.distance_to_finish_line(0);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    pawn,
                    &board.allowed_walls_for_pawn(pawn, &distances_to_finish).0,
                    0
                )
                .0,
            5
        );
    }

    #[test]
    fn test_relevant_unforceble_squares_should_exclude() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Vertical, Position { row: 0, col: 3 }),
            (WallDirection::Vertical, Position { row: 2, col: 3 }),
            (WallDirection::Vertical, Position { row: 4, col: 3 }),
            (WallDirection::Vertical, Position { row: 6, col: 3 }),
            (WallDirection::Horizontal, Position { row: 4, col: 4 }),
            (WallDirection::Horizontal, Position { row: 4, col: 6 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        {
            let pawn = &mut board.pawns[0];
            pawn.goal_row = 8;
            pawn.position = Position { row: 7, col: 5 };
        }
        let start = Instant::now();

        let distances_to_finish = board.distance_to_finish_line(0);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    board.pawns[0],
                    &board
                        .allowed_walls_for_pawn(board.pawns[0], &distances_to_finish)
                        .0,
                    0
                )
                .0,
            40,
        );
        println!("relevant squares {:?}", start.elapsed());

        let start = Instant::now();
        assert_eq!(
            board
                .test_version_find_all_squares_relevant_for_pawn(0)
                .number_of_squares,
            15
        );
        println!("relevant squares with excluding {:?}", start.elapsed());
    }

    #[test]
    fn test_relevant_unforceble_squares_should_not_exclude() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Vertical, Position { row: 1, col: 3 }),
            (WallDirection::Vertical, Position { row: 3, col: 3 }),
            (WallDirection::Vertical, Position { row: 5, col: 3 }),
            (WallDirection::Vertical, Position { row: 7, col: 3 }),
            (WallDirection::Horizontal, Position { row: 4, col: 4 }),
            (WallDirection::Horizontal, Position { row: 4, col: 6 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        {
            let pawn = &mut board.pawns[0];
            pawn.goal_row = 8;
            pawn.position = Position { row: 7, col: 5 };
        }
        let start = Instant::now();
        let distances_to_finish = board.distance_to_finish_line(0);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    board.pawns[0],
                    &board
                        .allowed_walls_for_pawn(board.pawns[0], &distances_to_finish)
                        .0,
                    0
                )
                .0,
            8 * 9,
        );
        println!("relevant squares {:?}", start.elapsed());

        let start = Instant::now();
        println!(
            "{:?}",
            board
                .test_version_find_all_squares_relevant_for_pawn(0)
                .number_of_squares
        );
        assert_eq!(
            board
                .test_version_find_all_squares_relevant_for_pawn(0)
                .number_of_squares,
            8 * 9,
        );
        println!("relevant squares with excluding {:?}", start.elapsed());

        println!("--------------------- FAILING TEST");
        board.place_wall(WallDirection::Horizontal, Position { row: 0, col: 0 });
        let start = Instant::now();
        let relevant_squares = board.test_version_find_all_squares_relevant_for_pawn(0);
        assert_eq!(relevant_squares.number_of_squares, 8 * 9 - 2);
        println!(
            "relevant squares with excluding {:?}, took: {:?}",
            relevant_squares,
            start.elapsed()
        );
    }

    #[test]
    fn test_relevant_unforceble_squares_unreachable_square() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Vertical, Position { row: 1, col: 3 }),
            (WallDirection::Vertical, Position { row: 3, col: 3 }),
            (WallDirection::Vertical, Position { row: 5, col: 3 }),
            (WallDirection::Vertical, Position { row: 7, col: 3 }),
            (WallDirection::Horizontal, Position { row: 4, col: 4 }),
            (WallDirection::Horizontal, Position { row: 4, col: 6 }),
            // Below is a litle unreachable square of size 4
            (WallDirection::Horizontal, Position { row: 1, col: 0 }),
            (WallDirection::Vertical, Position { row: 1, col: 1 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        {
            let pawn = &mut board.pawns[0];
            pawn.goal_row = 8;
            pawn.position = Position { row: 7, col: 5 };
        }
        let start = Instant::now();

        let distances_to_finish = board.distance_to_finish_line(0);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    board.pawns[0],
                    &board
                        .allowed_walls_for_pawn(board.pawns[0], &distances_to_finish)
                        .0,
                    0
                )
                .0,
            8 * 9,
        );
        println!("relevant squares {:?}", start.elapsed());

        let start = Instant::now();
        assert_eq!(
            board
                .test_version_find_all_squares_relevant_for_pawn(0)
                .number_of_squares,
            8 * 9 - 4
        );
        println!("relevant squares with excluding {:?}", start.elapsed());
    }

    #[test]
    fn test_relevant_unforceble_squares_back_square() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Horizontal, Position { row: 5, col: 3 }),
            (WallDirection::Horizontal, Position { row: 5, col: 5 }),
            (WallDirection::Horizontal, Position { row: 5, col: 7 }),
            (WallDirection::Vertical, Position { row: 6, col: 3 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        {
            let pawn = &mut board.pawns[1];
            pawn.goal_row = 0;
            pawn.position = Position { row: 8, col: 4 };
        }
        let start = Instant::now();
        let distances_to_finish = board.distance_to_finish_line(1);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    board.pawns[1],
                    &board
                        .allowed_walls_for_pawn(board.pawns[1], &distances_to_finish)
                        .0,
                    0
                )
                .0,
            8 * 9,
        );
        println!("relevant squares {:?}", start.elapsed());

        println!("{:?}", board.encode());

        let start = Instant::now();
        let rel_squares = board.test_version_find_all_squares_relevant_for_pawn(1);
        println!("relevant squares with excluding {:?}", start.elapsed());
        rel_squares.pretty_print();
        assert_eq!(rel_squares.number_of_squares, 8 * 9 - 15);
        assert_eq!(rel_squares.dist_walked_unhindered, 1);

        // Now we test the same on the right side
        let mut board = Board::new();

        let walls = [
            (WallDirection::Vertical, Position { row: 3, col: 3 }),
            (WallDirection::Vertical, Position { row: 5, col: 3 }),
            (WallDirection::Vertical, Position { row: 7, col: 3 }),
            (WallDirection::Horizontal, Position { row: 2, col: 4 }),
            (WallDirection::Horizontal, Position { row: 2, col: 6 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        {
            let pawn = &mut board.pawns[1];
            pawn.goal_row = 0;
            pawn.position = Position { row: 3, col: 8 };
        }

        let start = Instant::now();
        let distances_to_finish = board.distance_to_finish_line(1);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    board.pawns[1],
                    &board
                        .allowed_walls_for_pawn(board.pawns[1], &distances_to_finish)
                        .0,
                    0
                )
                .0,
            8 * 9,
        );
        println!(
            "relevant squares {:?} for board: {}",
            start.elapsed(),
            board.encode()
        );

        let start = Instant::now();
        let rel_squares = board.test_version_find_all_squares_relevant_for_pawn(1);
        println!(
            "relevant squares with excluding {:?} for board: {}",
            start.elapsed(),
            board.encode()
        );
        assert_eq!(rel_squares.number_of_squares, 8 * 9 - 30);
        assert_eq!(rel_squares.dist_walked_unhindered, 1);

        // Here we will place a wall behind the pawn.
        {
            let mut board = board.clone();
            {
                let pawn = &mut board.pawns[1];
                pawn.goal_row = 0;
                pawn.position = Position { row: 3, col: 7 };
            }
            // Now pawn can still be forced everywhere
            assert_eq!(
                board
                    .test_version_find_all_squares_relevant_for_pawn(1)
                    .number_of_squares,
                72
            );

            println!("{}", board.encode());
            // If we place another wall behind the pawn, it can again not be forced
            board.place_wall(WallDirection::Horizontal, Position { row: 3, col: 7 });
            let rel_squares = board.test_version_find_all_squares_relevant_for_pawn(1);
            assert_eq!(rel_squares.number_of_squares, 8 * 9 - 30);
            assert_eq!(rel_squares.dist_walked_unhindered, 2);
        }

        // Here we will place a wall two left of the pawn.
        {
            let mut board = board.clone();
            {
                let pawn = &mut board.pawns[1];
                pawn.goal_row = 0;
                pawn.position = Position { row: 4, col: 8 };
            }
            // Now pawn can still be forced everywhere
            assert_eq!(
                board
                    .test_version_find_all_squares_relevant_for_pawn(1)
                    .number_of_squares,
                72
            );

            // If we place another wall left of the pawn, it can again not be forced again, cause any block would block off whole path.
            // | --|   |
            // ||  |   |
            // ||  | 0 |
            board.place_wall(WallDirection::Vertical, Position { row: 3, col: 6 });
            println!("{}", board.encode());
            let rel_squares = board.test_version_find_all_squares_relevant_for_pawn(1);
            assert_eq!(rel_squares.number_of_squares, 8 * 9 - 30);
            assert_eq!(rel_squares.dist_walked_unhindered, 2);
        }
    }

    #[test]
    fn test_relevant_unforceble_squares_back_line() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Horizontal, Position { row: 6, col: 3 }),
            (WallDirection::Horizontal, Position { row: 6, col: 5 }),
            // Below is a litle unreachable square of size 4
            (WallDirection::Horizontal, Position { row: 6, col: 7 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }

        {
            let pawn = &mut board.pawns[0];
            pawn.goal_row = 8;
            pawn.position = Position { row: 7, col: 4 };
        }
        let start = Instant::now();
        let distances_to_finish = board.distance_to_finish_line(0);
        assert_eq!(
            board
                .open_routes
                .find_all_squares_relevant_for_pawn(
                    board.pawns[0],
                    &board
                        .allowed_walls_for_pawn(board.pawns[0], &distances_to_finish)
                        .0,
                    0
                )
                .0,
            8 * 9,
        );
        println!("relevant squares {:?}", start.elapsed());
        let rel_squares = board.test_version_find_all_squares_relevant_for_pawn(0);
        rel_squares.pretty_print();

        let start = Instant::now();
        println!("{}", board.encode());
        assert_eq!(
            board
                .test_version_find_all_squares_relevant_for_pawn(0)
                .number_of_squares,
            8 * 9
        );
        println!("relevant squares with excluding {:?}", start.elapsed());
    }

    #[test]
    fn test_double_step_cache_calc() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Horizontal, Position { row: 2, col: 4 }),
            (WallDirection::Vertical, Position { row: 2, col: 3 }),
            // Below is a litle unreachable square of size 4
            (WallDirection::Vertical, Position { row: 3, col: 4 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }
        board.pawns[0].position = Position { row: 3, col: 4 };
        board.pawns[1].position = Position { row: 4, col: 4 };
        board.turn = 0;
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        let next_move = Move::PawnMove(PawnMove::Up, Some(PawnMove::Up));
        let mut next_board = board.clone();
        next_board.game_move(next_move);
        let previous_move_cache = [
            cache[0].next_cache(next_move, &board, &next_board, 0),
            cache[1].next_cache(next_move, &board, &next_board, 1),
        ];
        let next_cache = [
            NextMovesCache::new(&next_board, 0),
            NextMovesCache::new(&next_board, 1),
        ];
        for i in 0..2 {
            assert_eq!(
                previous_move_cache[i].distances_to_finish,
                next_cache[i].distances_to_finish,
            );
            if previous_move_cache[i].allowed_walls_for_pawn != next_cache[i].allowed_walls_for_pawn
            {
                println!(
                    "allowed walls for pawn {}: {:?} are different",
                    i, next_board.pawns[i]
                );
                println!("previous_cache");
                previous_move_cache[i]
                    .allowed_walls_for_pawn
                    .pretty_print_wall();
                println!("new cache");
                next_cache[i].allowed_walls_for_pawn.pretty_print_wall();

                println!("walls");
                board.walls.pretty_print_wall();
            }
            assert_eq!(
                previous_move_cache[i].allowed_walls_for_pawn,
                next_cache[i].allowed_walls_for_pawn,
            );
            if previous_move_cache[i].relevant_squares != next_cache[i].relevant_squares {
                println!(
                    "allowed walls for pawn {}: {:?} are different",
                    i, next_board.pawns[i]
                );
                println!("previous_cache");
                previous_move_cache[i].relevant_squares.pretty_print();
                println!("new cache");
                next_cache[i].relevant_squares.pretty_print();

                println!("walls");
                board.walls.pretty_print_wall();
            }
            assert_eq!(
                previous_move_cache[i].relevant_squares,
                next_cache[i].relevant_squares,
            );
        }
    }

    #[test]
    fn test_close_goal_line() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Horizontal, Position { row: 1, col: 1 }),
            (WallDirection::Vertical, Position { row: 0, col: 3 }),
            // Below is a litle unreachable square of size 4
            (WallDirection::Vertical, Position { row: 2, col: 4 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }
        board.pawns[0].position = Position { row: 3, col: 4 };
        board.pawns[1].position = Position { row: 6, col: 3 };
        board.turn = 0;
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        let next_move = Move::Wall(WallDirection::Horizontal, Position { row: 0, col: 1 });
        let mut next_board = board.clone();
        next_board.game_move(next_move);
        let previous_move_cache = [
            cache[0].next_cache(next_move, &board, &next_board, 0),
            cache[1].next_cache(next_move, &board, &next_board, 1),
        ];
        let next_cache = [
            NextMovesCache::new(&next_board, 0),
            NextMovesCache::new(&next_board, 1),
        ];
        for i in 0..2 {
            assert_eq!(
                previous_move_cache[i].distances_to_finish,
                next_cache[i].distances_to_finish,
            );
            if previous_move_cache[i].allowed_walls_for_pawn != next_cache[i].allowed_walls_for_pawn
            {
                println!(
                    "allowed walls for pawn {}: {:?} are different",
                    i, next_board.pawns[i]
                );
                println!("previous_cache");
                previous_move_cache[i]
                    .allowed_walls_for_pawn
                    .pretty_print_wall();
                println!("new cache");
                next_cache[i].allowed_walls_for_pawn.pretty_print_wall();

                println!("walls");
                board.walls.pretty_print_wall();
            }
            assert_eq!(
                previous_move_cache[i].allowed_walls_for_pawn,
                next_cache[i].allowed_walls_for_pawn,
            );
            if previous_move_cache[i].relevant_squares != next_cache[i].relevant_squares {
                println!(
                    "allowed walls for pawn {}: {:?} are different",
                    i, next_board.pawns[i]
                );
                println!("previous_cache");
                previous_move_cache[i].relevant_squares.pretty_print();
                println!("new cache");
                next_cache[i].relevant_squares.pretty_print();

                println!("walls");
                board.walls.pretty_print_wall();
            }
            assert_eq!(
                previous_move_cache[i].relevant_squares,
                next_cache[i].relevant_squares,
            );
        }
    }

    #[test]
    fn test_walk_to_unhindered() {
        let mut board = Board::new();

        let walls = [
            (WallDirection::Horizontal, Position { row: 2, col: 3 }),
            (WallDirection::Horizontal, Position { row: 2, col: 5 }),
            (WallDirection::Horizontal, Position { row: 2, col: 7 }),
            (WallDirection::Vertical, Position { row: 3, col: 4 }),
            (WallDirection::Vertical, Position { row: 3, col: 5 }),
            (WallDirection::Vertical, Position { row: 4, col: 3 }),
            (WallDirection::Horizontal, Position { row: 5, col: 4 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }
        board.pawns[0].position = Position { row: 0, col: 0 };
        board.pawns[1].position = Position { row: 4, col: 5 };
        board.turn = 1;
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        println!("OLD BOARD");
        cache[1].relevant_squares.pretty_print();
        println!(
            "{}, {}",
            cache[1].relevant_squares.number_of_squares,
            cache[1].relevant_squares.dist_walked_unhindered
        );
        let next_move = Move::PawnMove(PawnMove::Up, None);
        let mut next_board = board.clone();
        next_board.game_move(next_move);
        let previous_move_cache = [
            cache[0].next_cache(next_move, &board, &next_board, 0),
            cache[1].next_cache(next_move, &board, &next_board, 1),
        ];
        let next_cache = [
            NextMovesCache::new(&next_board, 0),
            NextMovesCache::new(&next_board, 1),
        ];
        for i in 0..2 {
            assert_eq!(
                previous_move_cache[i].distances_to_finish,
                next_cache[i].distances_to_finish,
            );
            if previous_move_cache[i].allowed_walls_for_pawn != next_cache[i].allowed_walls_for_pawn
            {
                println!(
                    "allowed walls for pawn {}: {:?} are different",
                    i, next_board.pawns[i]
                );
                println!("previous_cache");
                previous_move_cache[i]
                    .allowed_walls_for_pawn
                    .pretty_print_wall();
                println!("new cache");
                next_cache[i].allowed_walls_for_pawn.pretty_print_wall();

                println!("walls");
                board.walls.pretty_print_wall();
            }
            assert_eq!(
                previous_move_cache[i].allowed_walls_for_pawn,
                next_cache[i].allowed_walls_for_pawn,
            );
            if previous_move_cache[i].relevant_squares != next_cache[i].relevant_squares {
                println!(
                    "allowed walls for pawn {}: {:?} are different",
                    i, next_board.pawns[i]
                );
                println!("previous_cache");
                previous_move_cache[i].relevant_squares.pretty_print();
                println!("new cache");
                next_cache[i].relevant_squares.pretty_print();

                println!("walls");
                board.walls.pretty_print_wall();
            }

            previous_move_cache[i].relevant_squares.pretty_print();

            next_cache[i].relevant_squares.pretty_print();
            println!("{}, move was {:?}", board.encode(), next_move);
            assert_eq!(
                previous_move_cache[i].relevant_squares,
                next_cache[i].relevant_squares,
            );
        }
    }

    #[test]
    fn test_blocked_off_pawn_0() {
        let mut board = Board::new();
        let walls = [
            (WallDirection::Horizontal, Position { row: 2, col: 3 }),
            (WallDirection::Horizontal, Position { row: 2, col: 5 }),
            (WallDirection::Horizontal, Position { row: 2, col: 7 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }
        board.pawns[0].position = Position { row: 3, col: 4 };
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        let pot_block_off = board.open_routes.test_check_lines(
            board.pawns[0],
            &cache[0].relevant_squares,
            cache[0].distances_to_finish,
        );
        assert_eq!(
            cache[0]
                .relevant_squares
                .number_of_left_relevant(pot_block_off),
            72 - 27
        );
        for row in pot_block_off {
            println!("{:?}", row);
        }

        let pot_block_off = board.open_routes.test_check_lines(
            board.pawns[1],
            &cache[1].relevant_squares,
            cache[1].distances_to_finish,
        );
        assert_eq!(
            cache[1]
                .relevant_squares
                .number_of_left_relevant(pot_block_off),
            72
        );
        for row in pot_block_off {
            println!("{:?}", row);
        }

        let walls = [
            (WallDirection::Vertical, Position { row: 3, col: 2 }),
            (WallDirection::Vertical, Position { row: 5, col: 2 }),
        ];

        for (dir, loc) in walls {
            println!("{}", board.place_wall(dir, loc));
        }
        let cache_new = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        let start = Instant::now();
        let pot_block_off = board.open_routes.test_check_lines(
            board.pawns[0],
            &cache_new[0].relevant_squares,
            cache[0].distances_to_finish,
        );
        println!("took: {:?}", start.elapsed());
        println!("------------------");

        board.walls.pretty_print_wall();
        for row in pot_block_off {
            println!("{:?}", row);
        }
        assert_eq!(
            cache[0]
                .relevant_squares
                .number_of_left_relevant(pot_block_off),
            72 - 27 - 9
        );
    }

    #[test]
    fn test_blocked_off_pawn_1() {
        let mut board = Board::new();
        let walls = [
            (WallDirection::Horizontal, Position { row: 5, col: 3 }),
            (WallDirection::Horizontal, Position { row: 5, col: 5 }),
            (WallDirection::Horizontal, Position { row: 5, col: 7 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }
        board.pawns[1].position = Position { row: 5, col: 4 };
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        let pot_block_off = board.open_routes.test_check_lines(
            board.pawns[1],
            &cache[1].relevant_squares,
            cache[1].distances_to_finish,
        );
        assert_eq!(
            cache[1]
                .relevant_squares
                .number_of_left_relevant(pot_block_off),
            72 - 27
        );
        for row in pot_block_off {
            println!("{:?}", row);
        }

        let pot_block_off = board.open_routes.test_check_lines(
            board.pawns[0],
            &cache[0].relevant_squares,
            cache[0].distances_to_finish,
        );
        println!("------");
        for row in pot_block_off {
            println!("{:?}", row);
        }
        assert_eq!(
            cache[0]
                .relevant_squares
                .number_of_left_relevant(pot_block_off),
            72
        );
        for row in pot_block_off {
            println!("{:?}", row);
        }

        let walls = [
            (WallDirection::Vertical, Position { row: 4, col: 2 }),
            (WallDirection::Vertical, Position { row: 2, col: 2 }),
        ];

        for (dir, loc) in walls {
            println!("{}", board.place_wall(dir, loc));
        }
        let cache_new = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        let start = Instant::now();
        let pot_block_off = board.open_routes.test_check_lines(
            board.pawns[1],
            &cache_new[1].relevant_squares,
            cache[1].distances_to_finish,
        );
        println!("took: {:?}", start.elapsed());
        println!("------");
        for row in pot_block_off {
            println!("{:?}", row);
        }

        assert_eq!(
            cache[1]
                .relevant_squares
                .number_of_left_relevant(pot_block_off),
            72 - 27 - 9
        );
        println!("------------------");
        board.walls.pretty_print_wall();
        for row in pot_block_off {
            println!("{:?}", row);
        }
    }

    #[test]
    fn test_blocked_off_side() {
        let mut board = Board::new();
        let walls = [
            (WallDirection::Vertical, Position { row: 0, col: 1 }),
            (WallDirection::Vertical, Position { row: 2, col: 1 }),
            (WallDirection::Vertical, Position { row: 4, col: 1 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }
        board.pawns[1].position = Position { row: 5, col: 4 };
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        let pot_block_off = board.open_routes.test_check_lines(
            board.pawns[0],
            &cache[0].relevant_squares,
            cache[0].distances_to_finish,
        );
        for row in pot_block_off {
            println!("{:?}", row);
        }

        assert_eq!(
            cache[0]
                .relevant_squares
                .number_of_left_relevant(pot_block_off),
            72 - 10
        );
    }
    #[test]
    fn test_blocked_off_side_4_length() {
        let mut board = Board::new();
        let walls = [
            (WallDirection::Vertical, Position { row: 0, col: 3 }),
            (WallDirection::Vertical, Position { row: 2, col: 3 }),
            (WallDirection::Vertical, Position { row: 4, col: 3 }),
        ];
        for (dir, loc) in walls {
            board.place_wall(dir, loc);
        }
        board.pawns[1].position = Position { row: 5, col: 4 };
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        let pot_block_off = board.open_routes.test_check_lines(
            board.pawns[0],
            &cache[0].relevant_squares,
            cache[0].distances_to_finish,
        );
        for row in pot_block_off {
            println!("{:?}", row);
        }

        assert_eq!(
            cache[0]
                .relevant_squares
                .number_of_left_relevant(pot_block_off),
            72 - 16
        );
    }

    #[test]
    fn test_relevant_squares_end() {
        let board = Board::decode("66;1B2;0A6;B1h;A2v;B3v;D3h;F3h;H3h;A4v;D4v;B5v;D5h;F5v;G5h;D6v;E6h;G6v;H6h;B7v;C8h;D8v").unwrap();

        let distances_to_finish =
            board.distance_to_finish_line(1).dist[board.pawns[1].position].unwrap();
        let cache = NextMovesCache::new(&board, 1);
        cache.relevant_squares.pretty_print();
        assert_eq!(
            cache.relevant_squares.dist_walked_unhindered,
            distances_to_finish
        );
    }
    /// New Test
    #[test]
    fn test_exclude_everyting_behind_pawn() {
        let prev_board = Board::decode(
            "40;2F7;1B3;A1h;C1v;D2h;F2h;H2h;C3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;H6h;A7h;C7h;E7h",
        )
        .unwrap();
        let old_cache = NextMovesCache::new(&prev_board, 0);

        let board = Board::decode(
            "41;2G7;1B3;A1h;C1v;D2h;F2h;H2h;C3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;H6h;A7h;C7h;E7h",
        )
        .unwrap();
        let start = Instant::now();
        NextMovesCache::new(&board, 0);
        println!("whole cache calc took: {:?}", start.elapsed());

        let start = Instant::now();
        old_cache.next_cache(
            Move::PawnMove(PawnMove::Right, None),
            &prev_board,
            &board,
            0,
        );
        println!("Using old cache took: {:?}", start.elapsed());

        // Number of squares should be 12, not 32....
        let start = Instant::now();
        let rel_squares = board.test_version_find_all_squares_relevant_for_pawn(0);
        println!("cache calc took: {:?}", start.elapsed());
        assert_eq!(rel_squares.number_of_squares, 12);
        let board = Board::decode(
            "41;2H7;1B3;A1h;C1v;D2h;F2h;H2h;C3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;H6h;A7h;C7h;E7h",
        )
        .unwrap();
        // Number of squares should be 12, not 32....
        let cache = NextMovesCache::new(&board, 0);
        assert_eq!(cache.relevant_squares.number_of_squares, 12);
    }

    #[test]
    fn test_excluding_pocket_two_sides() {
        let board = Board::decode(
            "36;1E5;1A4;A1h;C1v;D2h;F2h;H2h;A3h;C3v;F3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;F6h;A7h;C7h",
        )
        .unwrap();

        let cache = NextMovesCache::new(&board, 0);
        let next_board = Board::decode(
            "37;1F5;1A4;A1h;C1v;D2h;F2h;H2h;A3h;C3v;F3v;B4h;D4v;B5h;D5h;G5h;C6v;D6h;F6h;A7h;C7h",
        )
        .unwrap();

        let start = Instant::now();
        let prev_cache = cache.next_cache(
            Move::PawnMove(PawnMove::Right, None),
            &board,
            &next_board,
            0,
        );
        println!("from cache took: {:?}", start.elapsed());

        let next_cache = NextMovesCache::new(&next_board, 0);
        assert_eq!(prev_cache.relevant_squares, next_cache.relevant_squares);
        next_cache.relevant_squares.pretty_print();
    }

    #[test]
    fn test_excluding_double_jump() {
        let board = Board::decode(
            "20;1E4;1D4;F1h;H1h;A2h;C2h;D2v;E2h;F3v;A4h;D4h;G4h;B5h;D5h;E5v;F5h;H5h;D6v;F6v;D8v",
        )
        .unwrap();

        let start = Instant::now();
        let cache = NextMovesCache::new(&board, 0);
        println!("from scratchtook: {:?}", start.elapsed());

        let next_board = Board::decode(
            "20;1C4;1D4;F1h;H1h;A2h;C2h;D2v;E2h;F3v;A4h;D4h;G4h;B5h;D5h;E5v;F5h;H5h;D6v;F6v;D8v",
        )
        .unwrap();

        let start = Instant::now();
        let prev_cache = cache.next_cache(
            Move::PawnMove(PawnMove::Left, Some(PawnMove::Left)),
            &board,
            &next_board,
            0,
        );
        println!("from cache took: {:?}", start.elapsed());

        let start = Instant::now();
        let next_cache = NextMovesCache::new(&next_board, 0);
        println!("from scratchtook: {:?}", start.elapsed());

        prev_cache.relevant_squares.pretty_print();
        next_cache.relevant_squares.pretty_print();
        assert_eq!(prev_cache.relevant_squares, next_cache.relevant_squares);
    }

    #[test]
    fn test_wall_error() {
        let board = Board::decode("10;5E4;5E6;D1v;D2h;C3v;D3h;E5v").unwrap();

        let start = Instant::now();
        let cache = NextMovesCache::new(&board, 0);
        println!("from scratchtook: {:?}", start.elapsed());

        let next_board = Board::decode("11;4E4;5E6;D1v;D2h;C3v;D3h;E5v;D6h").unwrap();

        let start = Instant::now();

        let game_move = Move::Wall(WallDirection::Horizontal, Position { row: 5, col: 3 });
        let prev_cache = cache.next_cache(game_move, &board, &next_board, 0);
        println!("from cache took: {:?}", start.elapsed());

        let start = Instant::now();
        let next_cache = NextMovesCache::new(&next_board, 0);
        println!("from scratchtook: {:?}", start.elapsed());

        prev_cache.relevant_squares.pretty_print();
        next_cache.relevant_squares.pretty_print();

        assert_eq!(
            prev_cache.allowed_walls_for_pawn,
            next_cache.allowed_walls_for_pawn
        );
        assert_eq!(prev_cache.relevant_squares, next_cache.relevant_squares);

        // previous board: 10;5E4;5E6;D1v;D2h;C3v;D3h;E5v, next_board: 11;4E4;5E6;D1v;D2h;C3v;D3h;E5v;D6h, game_move:
    }

    #[test]
    fn test_only_bad_pawn_moves() {
        let board = Board::decode(
            "26;1E5;2F5;G2v;A3h;C3h;E3h;G3h;D4v;G4h;C5v;D5h;G5h;H5v;D6h;F6v;G6v;H6h;E7v;F8v",
        )
        .unwrap();

        let start = Instant::now();
        let cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        println!("from scratchtook: {:?}", start.elapsed());

        let next_moves = board.next_moves_with_scoring(true, &mut SmallRng::from_entropy(), &cache);
        println!("{:?}", next_moves);
        assert!(next_moves.into_iter().filter(|x| x.1 >= 0).count() >= 1);
    }

    fn compare_cache(
        previous_move_cache: [NextMovesCache; 2],
        next_cache: [NextMovesCache; 2],
        board: &Board,
        next_board: &Board,
        next_move: Move,
    ) {
        for i in 0..2 {
            assert_eq!(
                previous_move_cache[i].distances_to_finish,
                next_cache[i].distances_to_finish,
            );
            if previous_move_cache[i].allowed_walls_for_pawn != next_cache[i].allowed_walls_for_pawn
            {
                println!(
                    "allowed walls for pawn {}: {:?} are different",
                    i, next_board.pawns[i]
                );
                println!("previous_cache");
                previous_move_cache[i]
                    .allowed_walls_for_pawn
                    .pretty_print_wall();
                println!("new cache");
                next_cache[i].allowed_walls_for_pawn.pretty_print_wall();

                println!("walls");
                board.walls.pretty_print_wall();
            }
            assert_eq!(
                previous_move_cache[i].allowed_walls_for_pawn,
                next_cache[i].allowed_walls_for_pawn,
            );
            if previous_move_cache[i].relevant_squares != next_cache[i].relevant_squares {
                println!(
                    "allowed walls for pawn {}: {:?} are different",
                    i, next_board.pawns[i]
                );
                println!("previous_cache");
                previous_move_cache[i].relevant_squares.pretty_print();
                println!("new cache");
                next_cache[i].relevant_squares.pretty_print();

                println!("walls");
                board.walls.pretty_print_wall();
            }

            previous_move_cache[i].relevant_squares.pretty_print();

            next_cache[i].relevant_squares.pretty_print();
            println!("{}, move was {:?}", board.encode(), next_move);
            assert_eq!(
                previous_move_cache[i].relevant_squares,
                next_cache[i].relevant_squares,
            );
        }
    }
    // Need a way to get to previous board
    #[test]
    fn test_board_with_cache_error() {
        // 29;1E5;0H7;F1v;H2v;B3h;D3h;E3v;F3v;H3h;C4v;F4h;E5v;F5v;G5h;C6v;E6h;H6v;D7h;F7h;F8v;G8h Seems to be previous board
        let board = Board::decode(
            "30;1E5;0G6;F1v;H2v;B3h;D3h;E3v;F3v;H3h;C4v;F4h;E5v;F5v;G5h;C6v;E6h;H6v;D7h;F7h;F8v;G8h",
        )
        .unwrap();

        let start = Instant::now();
        let proper_cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        println!("from scratchtook: {:?}", start.elapsed());
        let longest_route = board.open_routes.furthest_walkable_unhindered(
            board.pawns[1],
            &proper_cache[1].allowed_walls_for_pawn,
            &proper_cache[1].distances_to_finish,
        );
        println!("LONGEST ROUTE: {:?}", longest_route);
        //panic!("STOP HERE FOR NOW");

        let next_moves =
            board.next_moves_with_scoring(true, &mut SmallRng::from_entropy(), &proper_cache);
        println!("{:?}", next_moves);
        assert!(next_moves.into_iter().filter(|x| x.1 >= 0).count() >= 1);
        for (prev_board, prev_move) in board.previous_boards() {
            println!("-----------------------");
            println!("{} {:?}", prev_board.encode(), prev_move);
            let cache = [
                NextMovesCache::new(&prev_board, 0),
                NextMovesCache::new(&prev_board, 1),
            ];
            let prev_cache = [
                cache[0].next_cache(prev_move, &prev_board, &board, 0),
                cache[1].next_cache(prev_move, &prev_board, &board, 1),
            ];
            if prev_move == Move::PawnMove(PawnMove::Left, None) {
                // This is a weird scenario, The pawn walks back towards an area that is excluded, but still nice to get it right
                //continue;
            }

            println!(
                "prev board moves: {:?}",
                prev_board.next_moves_with_scoring(true, &mut SmallRng::from_entropy(), &cache)
            );
            let next_moves_proper =
                board.next_moves_with_scoring(true, &mut SmallRng::from_entropy(), &proper_cache);
            println!("PROPER CACHE: {:?}", next_moves_proper);
            let next_moves_prev =
                board.next_moves_with_scoring(true, &mut SmallRng::from_entropy(), &prev_cache);
            println!("PREV CACHE: {:?}", next_moves_prev);
            println!("PRETTY PRINTING RELEVANT SQUARES PAWN 1 OLD BOARD");
            cache[1].relevant_squares.pretty_print();
            //assert_eq!(next_moves_proper, next_moves_prev);

            compare_cache(proper_cache, prev_cache, &prev_board, &board, prev_move);
        }
    }

    #[test]
    fn test_longest_route() {
        let board = Board::decode(
            "30;1E5;0G6;F1v;H2v;B3h;D3h;E3v;F3v;H3h;C4v;F4h;E5v;F5v;G5h;C6v;E6h;H6v;D7h;F7h;F8v;G8h",
        )
        .unwrap();

        let start = Instant::now();
        let proper_cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];
        println!("from scratchtook: {:?}", start.elapsed());
        let start = std::time::Instant::now();
        let mut longest_route = board.open_routes.furthest_walkable_unhindered(
            board.pawns[1],
            &proper_cache[1].allowed_walls_for_pawn,
            &proper_cache[1].distances_to_finish,
        );
        println!("{:?}", start.elapsed());
        assert_eq!(longest_route.pop().unwrap(), Position { row: 2, col: 6 });
    }

    #[test]
    fn test_wall_score() {
        let board = Board::decode("15;7F6;8D5;D3h;C6h;E6v;E8v;F5h").unwrap();

        let proper_cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        let interesting_wall = (WallDirection::Horizontal, Position { row: 4, col: 7 });
        println!(
            "SCORE WALL FOR WHITE PLAYER : {}",
            proper_cache[0].allowed_walls_for_pawn[interesting_wall].wall_score()
        );
        println!(
            "SCORE WALL FOR BLACK PLAYER : {}",
            proper_cache[1].allowed_walls_for_pawn[interesting_wall].wall_score()
        );
        // For player 0 this cuts of the whole back
        assert_eq!(
            proper_cache[0].allowed_walls_for_pawn[interesting_wall],
            WallType::Allowed(60)
        );

        // for the black player this is a pocket
        assert_eq!(
            proper_cache[1].allowed_walls_for_pawn[interesting_wall],
            WallType::Pocket
        );
    }
    #[test]
    fn test_wall_score_big_around_long() {
        let board = Board::decode("3;8E1;9E9;D3v;D5v;E6h").unwrap();

        let proper_cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        let interesting_wall = (WallDirection::Vertical, Position { row: 0, col: 3 });
        println!(
            "SCORE WALL FOR WHITE PLAYER : {}",
            proper_cache[0].allowed_walls_for_pawn[interesting_wall].wall_score()
        );
        println!(
            "SCORE WALL FOR BLACK PLAYER : {}",
            proper_cache[1].allowed_walls_for_pawn[interesting_wall].wall_score()
        );
        assert_eq!(
            proper_cache[0].allowed_walls_for_pawn[interesting_wall].wall_score(),
            16
        );

        assert_eq!(
            proper_cache[1].allowed_walls_for_pawn[interesting_wall].wall_score(),
            16
        );
    }

    #[test]
    fn test_wall_score_big_around_long_side_wall() {
        let board = Board::decode("9;5E1;6E9;H3v;C4h;E4h;G4h;C5h;E5h;G5h;B6v;H6v").unwrap();

        let proper_cache = [
            NextMovesCache::new(&board, 0),
            NextMovesCache::new(&board, 1),
        ];

        let interesting_wall = (WallDirection::Vertical, Position { row: 3, col: 3 });
        println!(
            "SCORE WALL FOR WHITE PLAYER : {}",
            proper_cache[0].allowed_walls_for_pawn[interesting_wall].wall_score()
        );
        println!(
            "SCORE WALL FOR BLACK PLAYER : {}",
            proper_cache[1].allowed_walls_for_pawn[interesting_wall].wall_score()
        );
        assert_eq!(
            proper_cache[0].allowed_walls_for_pawn[interesting_wall].wall_score(),
            19
        );

        assert_eq!(
            proper_cache[1].allowed_walls_for_pawn[interesting_wall].wall_score(),
            19
        );
    }
}
