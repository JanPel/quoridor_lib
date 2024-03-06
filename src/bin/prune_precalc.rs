use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use quoridor::*;

pub fn prune_hard(mc_node: &mut MCNode) {
    match mc_node {
        MCNode::Branch {
            move_options,
            scores_included,
            scores,
            ..
        } => {
            if scores.1 <= 1_500_000 {
                *move_options = None;
                *scores_included = 0;
            } else {
                if let Some(move_options) = move_options.as_mut() {
                    for (_, node, _) in move_options.iter_mut() {
                        prune_hard(node);
                    }
                }
            }
        }
        _ => (),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the first argument from the command line, which is the target directory
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <target_directory>", args[0]);
        std::process::exit(1);
    }
    let target_dir = &args[1];

    // Construct the path to the 'precalc' subdirectory
    let precalc_dir = Path::new(target_dir).join("precalc");

    // Iterate over all files in the precalc directory
    if precalc_dir.is_dir() {
        for entry in fs::read_dir(precalc_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Make sure we're dealing with a file
            if path.is_file() {
                println!("pruning {}", path.to_str().unwrap());
                // Attempt to deserialize the file into an MCNode
                let mut monte_carlo_tree =
                    MonteCarloTree::deserialize_file(&path.to_str().unwrap());
                prune_hard(&mut monte_carlo_tree.mc_node);
                monte_carlo_tree.serialize_to_file(path.to_str().unwrap());
            }
        }
    } else {
        eprintln!("The specified path is not a directory.");
        std::process::exit(1);
    }

    Ok(())
}
