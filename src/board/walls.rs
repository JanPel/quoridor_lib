use super::*;

#[derive(PartialEq, Eq, Clone, Copy, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WallDirection {
    Horizontal,
    Vertical,
}

impl WallDirection {
    pub fn orthogonal(&self) -> Self {
        match self {
            WallDirection::Horizontal => WallDirection::Vertical,
            WallDirection::Vertical => WallDirection::Horizontal,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WallType {
    // A wall that can not be physically placed
    Impossible,
    // A wall thats can not be placed, cause it blocks another panw
    Unallowed,
    Allowed(i8),
    Pocket,
}

impl WallType {
    pub fn is_allowed(&self) -> bool {
        match self {
            WallType::Allowed(_) => true,
            WallType::Pocket => true,
            _ => false,
        }
    }

    pub fn wall_score(&self) -> i8 {
        match self {
            WallType::Allowed(score) => *score,
            WallType::Pocket => 0,
            _ => -1,
        }
    }
}

#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Walls {
    pub horizontal: [[bool; 8]; 8],
    pub vertical: [[bool; 8]; 8],
}

impl Walls {
    pub fn mirror(&self) -> Self {
        let mut new_walls = Self::new();
        for row in 0..8 {
            for col in 0..8 {
                new_walls.horizontal[row][col] = self.horizontal[row][7 - col];
                new_walls.vertical[row][col] = self.vertical[row][7 - col];
            }
        }
        new_walls
    }
    pub fn new_allowed() -> Self {
        Self {
            horizontal: [[true; 8]; 8],
            vertical: [[true; 8]; 8],
        }
    }

    pub fn new() -> Self {
        Self {
            horizontal: [[false; 8]; 8],
            vertical: [[false; 8]; 8],
        }
    }
    pub fn update_allowed(&mut self, direction: WallDirection, location: Position) {
        let Position { col, row } = location;
        let row = row as usize;
        let col = col as usize;

        match direction {
            WallDirection::Horizontal => {
                if col >= 1 {
                    self.horizontal[row][col - 1] = false;
                }
                self.horizontal[row][col] = false;
                if col <= 6 {
                    self.horizontal[row][col + 1] = false;
                }
                self.vertical[row][col] = false;
            }
            WallDirection::Vertical => {
                if row >= 1 {
                    self.vertical[row - 1][col] = false;
                }
                self.vertical[row][col] = false;
                if row <= 6 {
                    self.vertical[row + 1][col] = false;
                }
                self.horizontal[row][col] = false;
            }
        }
    }

    // This function should print the walls, first we wanna print the horizontal ones and then the vertical ones. We dont' wanna print true and false
    // But rather "X" if a position contains a wall and " " if it doesn't. For efficiency lets write to a buffer and then print the whole buffer
    pub fn pretty_print_wall(&self) {
        let mut buffer = String::new();
        for row in 0..8 {
            buffer.push_str(&format!("{} HOR ", row));
            for col in 0..8 {
                buffer.push_str("|");
                if self.horizontal[row][col] {
                    buffer.push_str("X");
                } else {
                    buffer.push_str(" ");
                }
                buffer.push_str("|");
            }
            buffer.push_str("\n");
            buffer.push_str(&format!("{} VER ", row));
            for col in 0..8 {
                buffer.push_str("|");
                if self.vertical[row][col] {
                    buffer.push_str("X");
                } else {
                    buffer.push_str(" ");
                }
                buffer.push_str("|");
            }
            buffer.push_str("\n");
        }
        println!("{}", buffer);
    }

    // places a wall on the board, by setting the location to true.
    pub fn place_wall(&mut self, direction: WallDirection, location: Position) {
        let Position { col, row } = location;
        let row = row as usize;
        let col = col as usize;
        match direction {
            WallDirection::Horizontal => {
                self.horizontal[row][col] = true;
            }
            WallDirection::Vertical => {
                self.vertical[row][col] = true;
            }
        }
    }

    pub fn is_allowed(&self, direction: WallDirection, location: Position) -> bool {
        let Position { col, row } = location;
        let row = row as usize;
        let col = col as usize;
        match direction {
            WallDirection::Horizontal => self.horizontal[row][col],
            WallDirection::Vertical => self.vertical[row][col],
        }
    }

    pub fn wall_not_allowed(&mut self, direction: WallDirection, location: Position) {
        let Position { col, row } = location;
        let row = row as usize;
        let col = col as usize;
        match direction {
            WallDirection::Horizontal => self.horizontal[row][col] = false,
            WallDirection::Vertical => self.vertical[row][col] = false,
        }
    }

    /// Should update allowed walls, If we place a wall. THe walls next to it are not allowed anymore, cause walls have size 2.
    /// Also the wall crossing this wall is not allowed anymore. So if we have a horizontal wall, the verical walls with the same coordinates is not allwed anymore.

    pub fn connected_points(
        &self,
        direction: WallDirection,
        location: Position,
        // We can add this paramater to add an extra check for vertical rows in the second to last row.
        goal_row: Option<i8>,
    ) -> (bool, bool, bool, bool) {
        let Position { row, col } = location;
        let row = row as usize;
        let col = col as usize;
        match direction {
            WallDirection::Horizontal => {
                let mut left = col == 0;
                if col >= 1 {
                    if col >= 2 {
                        left = left || self.horizontal[row][col - 2];
                    }
                    left = left || self.vertical[row][col - 1];

                    if row >= 1 {
                        left = left || self.vertical[row - 1][col - 1];
                    }
                    if row <= 6 {
                        left = left || (self.vertical[row + 1][col - 1]);
                    }
                }

                let mut middle_top = false;

                let mut middle_bottom = false;

                if row >= 1 {
                    middle_top = self.vertical[row - 1][col];
                }
                if row <= 6 {
                    middle_bottom = self.vertical[row + 1][col];
                }

                let mut right = col == 7;
                if col <= 6 {
                    if col <= 5 {
                        right = right || self.horizontal[row][col + 2];
                    }
                    right = right || self.vertical[row][col + 1];

                    if row >= 1 {
                        right = right || self.vertical[row - 1][col + 1];
                    }
                    if row <= 6 {
                        right = right || self.vertical[row + 1][col + 1];
                    }
                }
                (left, middle_top, middle_bottom, right)
            }
            WallDirection::Vertical => {
                let mut down = row == 0;
                if let Some(0) = goal_row {
                    down = down || row == 1;
                }
                if row >= 1 {
                    if row >= 2 {
                        down = down || self.vertical[row - 2][col];
                    }
                    down = down || self.horizontal[row - 1][col];

                    if col >= 1 {
                        down = down || self.horizontal[row - 1][col - 1];
                    }
                    if col <= 6 {
                        down = down || self.horizontal[row - 1][col + 1];
                    }
                }

                let mut middle_top = false;
                let mut middle_bottom = false;

                if col >= 1 {
                    middle_top = self.horizontal[row][col - 1];
                }
                if col <= 6 {
                    middle_bottom = self.horizontal[row][col + 1];
                }

                let mut up = row == 7;
                if let Some(8) = goal_row {
                    up = up || row == 6;
                }

                if row <= 6 {
                    if row <= 5 {
                        up = up || self.vertical[row + 2][col];
                    }
                    up = up || self.horizontal[row + 1][col];

                    if col >= 1 {
                        up = up || self.horizontal[row + 1][col - 1];
                    }
                    if col <= 6 {
                        up = up || self.horizontal[row + 1][col + 1];
                    }
                }
                (down, middle_top, middle_bottom, up)
            }
        }
    }

    pub fn sides_of_wall(
        &self,
        direction: WallDirection,
        location: Position,
    ) -> Option<[[Position; 2]; 2]> {
        let (left, middle_top, middle_bottom, right) =
            self.connected_points(direction, location, None);
        if left as u8 + (middle_top || middle_bottom) as u8 + right as u8 <= 1 {
            return None;
        }
        if left && !(middle_top || middle_bottom) && right {
            return Some(positions_same_side_of_wall(direction, location));
        }
        let single_index = if left && middle_top && !middle_bottom && !right {
            (0, 0)
        } else if left && !middle_top && middle_bottom && !right {
            match direction {
                WallDirection::Horizontal => (1, 0),
                WallDirection::Vertical => (0, 1),
            }
        } else if left && middle_top && middle_bottom && !right {
            // for test
            // Special case
            //return early
            // we need to know middle top or middle bottom
            return match direction {
                WallDirection::Horizontal => {
                    let top = Position {
                        row: location.row,
                        col: location.col,
                    };
                    let bottom = Position {
                        row: location.row + 1,
                        col: location.col,
                    };
                    Some([[top, top], [bottom, bottom]])
                }
                WallDirection::Vertical => {
                    let left = Position {
                        row: location.row,
                        col: location.col,
                    };
                    let right = Position {
                        row: location.row,
                        col: location.col + 1,
                    };
                    Some([[left, left], [right, right]])
                }
            };
        } else if !left && middle_top && !middle_bottom && right {
            match direction {
                WallDirection::Horizontal => (0, 1),
                WallDirection::Vertical => (1, 0),
            }
        } else if !left && !middle_top && middle_bottom && right {
            (1, 1)
        } else if !left && middle_top && middle_bottom && right {
            return match direction {
                WallDirection::Horizontal => {
                    let top = Position {
                        row: location.row,
                        col: location.col + 1,
                    };
                    let bottom = Position {
                        row: location.row + 1,
                        col: location.col + 1,
                    };
                    Some([[top, top], [bottom, bottom]])
                }
                WallDirection::Vertical => {
                    let left = Position {
                        row: location.row + 1,
                        col: location.col,
                    };
                    let right = Position {
                        row: location.row + 1,
                        col: location.col + 1,
                    };
                    Some([[left, left], [right, right]])
                }
            };
        } else {
            return None;
        };

        let other_side_0 = ((single_index.0 + 1) % 2, single_index.1);
        let other_side_1 = (single_index.0, (single_index.1 + 1) % 2);

        let single_pos = Position {
            row: location.row + single_index.0,
            col: location.col + single_index.1,
        };

        let other_pos_0 = Position {
            row: location.row + other_side_0.0,
            col: location.col + other_side_0.1,
        };
        let other_pos_1 = Position {
            row: location.row + other_side_1.0,
            col: location.col + other_side_1.1,
        };

        return Some([[single_pos, single_pos], [other_pos_0, other_pos_1]]);
    }

    pub fn connect_on_two_points(
        &self,
        direction: WallDirection,
        location: Position,
        // We can add this paramater to add an extra check for vertical rows in the second to last row.
        goal_row: Option<i8>,
    ) -> bool {
        let (left, middle_top, middle_bottom, right) =
            self.connected_points(direction, location, goal_row);
        left as i8 + (middle_top || middle_bottom) as i8 + right as i8 >= 2
    }

    // Here we check whether a wall is connected to two or more other walls. If this is not the case, a wall will never block of a path totally.
    pub fn number_of_walls_wall_connect(&self, direction: WallDirection, location: Position) -> i8 {
        let mut count: i8 = 0;
        let Position { row, col } = location;
        let row = row as usize;
        let col = col as usize;
        match direction {
            WallDirection::Horizontal => {
                // first we check adjacent horizontal walls, or next to edge of board (parrallel)
                // we dont' want to include the walls on the edge of the board
                if col == 0 || col == 7 {
                    count += 1;
                }
                if col >= 2 {
                    count += self.horizontal[row][col - 2] as i8;
                }
                if col <= 5 {
                    count += self.horizontal[row][col + 2] as i8;
                }

                // Now we will check adjacent vertical walls (orthogonal)
                // T shaped walls
                if col >= 1 {
                    count += self.vertical[row][col - 1] as i8;
                }
                if col <= 6 {
                    count += self.vertical[row][col + 1] as i8;
                }

                if row >= 1 {
                    count += self.vertical[row - 1][col] as i8;
                    count += (col < 7 && self.vertical[row - 1][col + 1]) as i8;
                    count += (col >= 1 && self.vertical[row - 1][col - 1]) as i8;
                }

                if row < 7 {
                    count += self.vertical[row + 1][col] as i8;
                    count += (col < 7 && self.vertical[row + 1][col + 1]) as i8;
                    count += (col >= 1 && self.vertical[row + 1][col - 1]) as i8;
                }
            }
            WallDirection::Vertical => {
                // first we check adjacent vertical walls, or next to edge of board (parrallel)
                // We don't want to include vertical walls on the back lines
                //if row == 0 || row == 7 {
                //    count += 1;
                //}
                if row >= 2 {
                    count += self.vertical[row - 2][col] as i8;
                }
                if row <= 5 {
                    count += self.vertical[row + 2][col] as i8;
                }

                // now we will adjacent horizontal walls (orthogonal)
                // T shaped walls
                if row >= 1 {
                    count += self.horizontal[row - 1][col] as i8;
                }
                if row <= 6 {
                    count += self.horizontal[row + 1][col] as i8;
                }
                if col >= 1 {
                    count += self.horizontal[row][col - 1] as i8;
                    count += (row < 7 && self.horizontal[row + 1][col - 1]) as i8;
                    count += (row >= 1 && self.horizontal[row - 1][col - 1]) as i8;
                }
                if col < 7 {
                    count += self.horizontal[row][col + 1] as i8;
                    count += (row < 7 && self.horizontal[row + 1][col + 1]) as i8;
                    count += (row >= 1 && self.horizontal[row - 1][col + 1]) as i8;
                }
            }
        };
        count
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WallEffects {
    pub horizontal: [[Option<WallEffect>; 8]; 8],
    pub vertical: [[Option<WallEffect>; 8]; 8],
}

impl WallEffects {
    pub fn new() -> Self {
        Self {
            horizontal: [[None; 8]; 8],
            vertical: [[None; 8]; 8],
        }
    }
    pub fn new_allowed_with_score(
        &self,
        rel_squares: RelevantSquares,
        mut old_allowed: AllowedWalls,
    ) -> AllowedWalls {
        for row in 0..8 {
            for col in 0..8 {
                for dir in [WallDirection::Horizontal, WallDirection::Vertical] {
                    let pos = Position {
                        row: row as i8,
                        col: col as i8,
                    };
                    let wall = (dir, pos);
                    let effect = &self[wall];
                    if let Some(effect) = effect.as_ref() {
                        match &mut old_allowed[wall] {
                            WallType::Allowed(score) => {
                                *score = effect.wall_score(rel_squares);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        old_allowed
    }
}

impl std::ops::Index<(WallDirection, Position)> for WallEffects {
    type Output = Option<WallEffect>;

    fn index(&self, index: (WallDirection, Position)) -> &Self::Output {
        let (wall_direction, pos) = index;
        match wall_direction {
            WallDirection::Horizontal => &self.horizontal[pos],
            WallDirection::Vertical => &self.vertical[pos],
        }
    }
}

impl std::ops::IndexMut<(WallDirection, Position)> for WallEffects {
    fn index_mut(&mut self, index: (WallDirection, Position)) -> &mut Self::Output {
        let (wall_direction, pos) = index;
        match wall_direction {
            WallDirection::Horizontal => &mut self.horizontal[pos],
            WallDirection::Vertical => &mut self.vertical[pos],
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AllowedWalls {
    pub horizontal: [[WallType; 8]; 8],
    pub vertical: [[WallType; 8]; 8],
}

impl std::ops::Index<(WallDirection, Position)> for AllowedWalls {
    type Output = WallType;

    fn index(&self, index: (WallDirection, Position)) -> &Self::Output {
        let (wall_direction, pos) = index;
        match wall_direction {
            WallDirection::Horizontal => &self.horizontal[pos],
            WallDirection::Vertical => &self.vertical[pos],
        }
    }
}

impl std::ops::IndexMut<(WallDirection, Position)> for AllowedWalls {
    fn index_mut(&mut self, index: (WallDirection, Position)) -> &mut Self::Output {
        let (wall_direction, pos) = index;
        match wall_direction {
            WallDirection::Horizontal => &mut self.horizontal[pos],
            WallDirection::Vertical => &mut self.vertical[pos],
        }
    }
}

impl AllowedWalls {
    pub const fn new() -> Self {
        Self {
            horizontal: [[WallType::Allowed(0); 8]; 8],
            vertical: [[WallType::Allowed(0); 8]; 8],
        }
    }
    pub const fn zero() -> Self {
        Self {
            horizontal: [[WallType::Allowed(0); 8]; 8],
            vertical: [[WallType::Allowed(0); 8]; 8],
        }
    }
    pub fn pretty_print_wall(&self) {
        let mut buffer = String::new();
        for row in 0..8 {
            buffer.push_str(&format!("{} HOR ", row));
            for col in 0..8 {
                buffer.push_str("|");
                if self.horizontal[row][col].is_allowed() {
                    buffer.push_str("X");
                } else {
                    buffer.push_str(" ");
                }
            }
            buffer.push_str("\n");
            buffer.push_str(&format!("{} VER ", row));
            for col in 0..8 {
                buffer.push_str("|");
                if self.vertical[row][col].is_allowed() {
                    buffer.push_str("X");
                } else {
                    buffer.push_str(" ");
                }
            }
            buffer.push_str("\n");
        }
        println!("{}", buffer);
    }
    fn update_allowed(&mut self, direction: WallDirection, location: Position) {
        let Position { col, row } = location;
        let row = row as usize;
        let col = col as usize;

        match direction {
            WallDirection::Horizontal => {
                if col >= 1 {
                    self.horizontal[row][col - 1] = WallType::Impossible;
                }
                self.horizontal[row][col] = WallType::Impossible;
                if col <= 6 {
                    self.horizontal[row][col + 1] = WallType::Impossible;
                }
                self.vertical[row][col] = WallType::Impossible;
            }
            WallDirection::Vertical => {
                if row >= 1 {
                    self.vertical[row - 1][col] = WallType::Impossible;
                }
                self.vertical[row][col] = WallType::Impossible;
                if row <= 6 {
                    self.vertical[row + 1][col] = WallType::Impossible;
                }
                self.horizontal[row][col] = WallType::Impossible;
            }
        }
    }

    fn get_adjacent_walls(
        &self,
        dir: WallDirection,
        pos: Position,
    ) -> Vec<(WallDirection, Position)> {
        let mut adjacent_walls = Vec::new();
        let Position { row, col } = pos;
        let row = row as usize;
        let col = col as usize;

        match dir {
            WallDirection::Horizontal => {
                if col >= 2 && self.horizontal[row][col - 2].is_allowed() {
                    adjacent_walls.push((
                        WallDirection::Horizontal,
                        Position {
                            row: row as i8,
                            col: (col - 2) as i8,
                        },
                    ));
                }
                if col <= 5 && self.horizontal[row][col + 2].is_allowed() {
                    adjacent_walls.push((
                        WallDirection::Horizontal,
                        Position {
                            row: row as i8,
                            col: (col + 2) as i8,
                        },
                    ));
                }

                if col >= 1 && self.vertical[row][col - 1].is_allowed() {
                    adjacent_walls.push((
                        WallDirection::Vertical,
                        Position {
                            row: row as i8,
                            col: (col - 1) as i8,
                        },
                    ));
                }
                if col <= 6 && self.vertical[row][col + 1].is_allowed() {
                    adjacent_walls.push((
                        WallDirection::Vertical,
                        Position {
                            row: row as i8,
                            col: (col + 1) as i8,
                        },
                    ));
                }

                if row >= 1 {
                    if self.vertical[row - 1][col].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Vertical,
                            Position {
                                row: (row - 1) as i8,
                                col: col as i8,
                            },
                        ));
                    }
                    if col < 7 && self.vertical[row - 1][col + 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Vertical,
                            Position {
                                row: (row - 1) as i8,
                                col: (col + 1) as i8,
                            },
                        ));
                    }
                    if col >= 1 && self.vertical[row - 1][col - 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Vertical,
                            Position {
                                row: (row - 1) as i8,
                                col: (col - 1) as i8,
                            },
                        ));
                    }
                }

                if row < 7 {
                    if self.vertical[row + 1][col].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Vertical,
                            Position {
                                row: (row + 1) as i8,
                                col: col as i8,
                            },
                        ));
                    }
                    if col < 7 && self.vertical[row + 1][col + 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Vertical,
                            Position {
                                row: (row + 1) as i8,
                                col: (col + 1) as i8,
                            },
                        ));
                    }
                    if col >= 1 && self.vertical[row + 1][col - 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Vertical,
                            Position {
                                row: (row + 1) as i8,
                                col: (col - 1) as i8,
                            },
                        ));
                    }
                }
            }
            WallDirection::Vertical => {
                if row >= 2 && self.vertical[row - 2][col].is_allowed() {
                    adjacent_walls.push((
                        WallDirection::Vertical,
                        Position {
                            row: (row - 2) as i8,
                            col: col as i8,
                        },
                    ));
                }
                if row <= 5 && self.vertical[row + 2][col].is_allowed() {
                    adjacent_walls.push((
                        WallDirection::Vertical,
                        Position {
                            row: (row + 2) as i8,
                            col: col as i8,
                        },
                    ));
                }

                if row >= 1 && self.horizontal[row - 1][col].is_allowed() {
                    adjacent_walls.push((
                        WallDirection::Horizontal,
                        Position {
                            row: (row - 1) as i8,
                            col: col as i8,
                        },
                    ));
                }
                if row <= 6 && self.horizontal[row + 1][col].is_allowed() {
                    adjacent_walls.push((
                        WallDirection::Horizontal,
                        Position {
                            row: (row + 1) as i8,
                            col: col as i8,
                        },
                    ));
                }
                if col >= 1 {
                    if self.horizontal[row][col - 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Horizontal,
                            Position {
                                row: row as i8,
                                col: (col - 1) as i8,
                            },
                        ));
                    }
                    if row < 7 && self.horizontal[row + 1][col - 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Horizontal,
                            Position {
                                row: (row + 1) as i8,
                                col: (col - 1) as i8,
                            },
                        ));
                    }
                    if row >= 1 && self.horizontal[row - 1][col - 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Horizontal,
                            Position {
                                row: (row - 1) as i8,
                                col: (col - 1) as i8,
                            },
                        ));
                    }
                }
                if col < 7 {
                    if self.horizontal[row][col + 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Horizontal,
                            Position {
                                row: row as i8,
                                col: (col + 1) as i8,
                            },
                        ));
                    }
                    if row < 7 && self.horizontal[row + 1][col + 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Horizontal,
                            Position {
                                row: (row + 1) as i8,
                                col: (col + 1) as i8,
                            },
                        ));
                    }
                    if row >= 1 && self.horizontal[row - 1][col + 1].is_allowed() {
                        adjacent_walls.push((
                            WallDirection::Horizontal,
                            Position {
                                row: (row - 1) as i8,
                                col: (col + 1) as i8,
                            },
                        ));
                    }
                }
            }
        }

        adjacent_walls
    }

    pub fn calculate_new_cache(
        mut self,
        game_move: Move,
        // This is the position of where the pawn used to be.
        mut pawn: Pawn,
        // This variable tells us whether the pawn for whom these walls are cached made this move.
        did_cache_pawn_move: bool,
        new_board: &Board,
        distances: &DistancesToFinish,
    ) -> (Self, WallEffects) {
        let mut wall_effects = WallEffects::new();
        match game_move {
            Move::Wall(dir, pos) => {
                if !new_board.walls.connect_on_two_points(dir, pos, None) {
                    // In this case we only need to check the walls that touch this wall. Cause this wall didn't block off any paths to the finish.
                    for wall in self.get_adjacent_walls(dir, pos) {
                        let (allowed, effect) =
                            new_board.is_wall_allowed_pawn_new(wall, pawn, distances);
                        self[wall] = allowed;
                        wall_effects[wall] = effect;
                    }
                    self.update_allowed(dir, pos);
                    return (self, wall_effects);
                };
                new_board.allowed_walls_for_pawn(pawn, distances)
            }
            Move::PawnMove(first_step, second_step) => {
                if !did_cache_pawn_move {
                    return (self, wall_effects);
                }
                let old_position = pawn.position;
                pawn.position = pawn.position.add_pawn_moves(first_step, second_step);
                for wall in wall_across_path(old_position, first_step, second_step) {
                    let (allowed, effect) =
                        new_board.is_wall_allowed_pawn_new(wall, pawn, distances);
                    self[wall] = allowed;
                    wall_effects[wall] = effect;
                }
                (self, wall_effects)
            }
        }
    }
}
