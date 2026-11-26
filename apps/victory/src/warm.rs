//! `victory warm`: host pipelines made ahead of a run (`known_pipelines`), in
//! processes of their own: a compiler that crashes or hangs costs one pipeline.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use victory_hle::known_pipelines as known;

/// Pipelines a process makes before another takes its place: the compiler's
/// and the translations' memory goes with it.
const AT_ONCE: usize = 40;
/// A pipeline that takes longer is taken for one the compiler hangs on.
const LONGEST: Duration = Duration::from_secs(180);

/// As the runs count one that takes long.
const SLOW_MS: f64 = 1000.0;

fn folder() -> Result<PathBuf> {
    let cfg = victory_core::Config::load()?;
    Ok(cfg.cache_subdir("shaders")?.with_file_name("pipelines"))
}

pub fn run(jobs: Option<usize>, limit: Option<usize>, again: bool, used: bool, these: Option<String>) -> Result<()> {
    let dir = folder()?;
    if let Some(these) = these {
        return make(&dir, &these);
    }
    known::use_translations(dir.with_file_name("shaders"));
    warm(&dir, &known::fingerprint(), jobs, limit, again, used, start_worker)
}

fn warm(
    dir: &Path,
    fingerprint: &str,
    jobs: Option<usize>,
    limit: Option<usize>,
    again: bool,
    used: bool,
    start_worker: impl Fn(&[usize]) -> Result<Child> + Sync,
) -> Result<()> {
    let began = Instant::now();
    let done_file = dir.join("warm-done.txt");
    let bad_file = dir.join("warm-bad.txt");
    let keys = known::keys(dir);
    let listed = std::fs::read_to_string(&done_file).unwrap_or_default();
    // Old "<fingerprint> <key>" records included refusals. Only explicit
    // outcomes count, and a later --again attempt supersedes an earlier one.
    let mut history = HashMap::new();
    for (attempt, line) in listed.lines().enumerate() {
        let mut fields = line.split(' ');
        if fields.next() == Some(fingerprint)
            && let (Some(status @ ("made" | "refused")), Some(key), None) =
                (fields.next(), fields.next(), fields.next())
        {
            history.insert(key, (status == "made", attempt));
        }
    }
    let bad = std::fs::read_to_string(&bad_file).unwrap_or_default();
    let bad: HashSet<&str> = bad.lines().filter_map(|l| l.split(' ').next()).collect();
    let early = std::fs::read_to_string(dir.join("early-pipelines.txt")).unwrap_or_default();
    let early: HashSet<&str> = early.lines().collect();
    let mut all: Vec<usize> = (0..keys.len())
        .filter(|&n| {
            let key = keys[n].as_str();
            (again || !matches!(history.get(key), Some((true, _)))) && !bad.contains(key) && (!used || early.contains(key))
        })
        .collect();
    // Untried keys first (used ones first among those), then the oldest
    // attempts. Even repeated --limit 1 runs reach every persistent refusal.
    all.sort_by_key(|&n| {
        let key = keys[n].as_str();
        (history.get(key).map(|&(_, attempt)| attempt), !early.contains(key), key)
    });
    all.truncate(limit.unwrap_or(usize::MAX));
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let jobs = jobs.unwrap_or(cores * 2 / 3).max(1);
    println!(
        "{} pipelines known, {} made before ({fingerprint}), {} bad; {} to make, {jobs} processes",
        keys.len(),
        history.values().filter(|(made, _)| *made).count(),
        bad.len(),
        all.len()
    );
    let state = Mutex::new((Vec::<f64>::new(), 0usize, 0usize));
    let done_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&done_file)?);
    // (Read back by the runs: these are made beside the others, not before.)
    let slow_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(dir.join("slow-pipelines.txt"))?);
    let bad_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&bad_file)?);
    run_batches(
        &all,
        jobs,
        start_worker,
        |n, ms, there| {
            if ms >= SLOW_MS {
                let _ = writeln!(lock(&slow_out), "\n{} {ms:.0}", keys[n]);
            }
            let mut state = lock(&state);
            state.0.push(ms);
            if !there {
                if state.1 == 0 {
                    println!("pipeline {n} {} has no host pipeline. Rerun `victory warm --these {n}` for worker diagnostics.", keys[n]);
                }
                state.1 += 1;
            }
            let count = state.0.len();
            if count.is_multiple_of(1000) {
                println!("{count} of {}, {:.0} s", all.len(), began.elapsed().as_secs_f64());
            }
            drop(state);
            let status = if there { "made" } else { "refused" };
            let _ = writeln!(lock(&done_out), "{fingerprint} {status} {}", keys[n]);
        },
        |n, why| {
            println!("pipeline {n} {}: {why}", keys[n]);
            let _ = writeln!(lock(&bad_out), "{} {why}", keys[n]);
            lock(&state).2 += 1;
        },
    )?;
    let (mut times, refused, crashed) = state.into_inner().unwrap_or_else(|e| e.into_inner());
    times.sort_by(f64::total_cmp);
    let at = |q: f64| times.get((times.len() as f64 * q) as usize).copied().unwrap_or(0.0);
    println!(
        "{} made in {:.0} s, {refused} have no host pipeline, {crashed} ended their process; ms per reported attempt: median {:.1}, p90 {:.1}, p99 {:.1}, most {:.0}, sum {:.0} s",
        times.len() - refused,
        began.elapsed().as_secs_f64(),
        at(0.5),
        at(0.9),
        at(0.99),
        times.last().copied().unwrap_or(0.0),
        times.iter().sum::<f64>() / 1e3
    );
    Ok(())
}

fn run_batches(
    all: &[usize],
    jobs: usize,
    start: impl Fn(&[usize]) -> Result<Child> + Sync,
    each: impl Fn(usize, f64, bool) + Sync,
    bad: impl Fn(usize, String) + Sync,
) -> Result<()> {
    let chunks: Vec<&[usize]> = all.chunks(AT_ONCE).collect();
    let next = AtomicUsize::new(0);
    // Stop launching workers after a setup failure. In-flight workers can still report.
    let failure = OnceLock::new();
    std::thread::scope(|s| {
        for _ in 0..jobs.min(chunks.len()) {
            s.spawn(|| {
                while let Some(&chunk) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let mut rest = chunk;
                    while !rest.is_empty() {
                        if failure.get().is_some() {
                            return;
                        }
                        let outcome = start(rest)
                            .context("could not start warm worker")
                            .and_then(|process| child(process, &each));
                        match outcome {
                            Ok(Some((made, why))) => {
                                let remaining = &rest[made.min(rest.len())..];
                                let Some((&n, after)) = remaining.split_first() else {
                                    break;
                                };
                                bad(n, why);
                                rest = after;
                            }
                            Ok(None) => break,
                            Err(error) => {
                                let _ = failure.set(error);
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
    failure.into_inner().map_or(Ok(()), Err)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn start_worker(these: &[usize]) -> Result<Child> {
    use std::os::windows::process::CommandExt;
    const IDLE_PRIORITY_CLASS: u32 = 0x40;
    let exe = std::env::current_exe().context("could not locate the warm worker executable")?;
    let list: Vec<String> = these.iter().map(usize::to_string).collect();
    Command::new(&exe)
        .args(["warm", "--these", &list.join(",")])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(IDLE_PRIORITY_CLASS)
        .spawn()
        .with_context(|| format!("could not spawn {}", exe.display()))
}

/// On abnormal exit, return the number of reports before the failure and the exit reason.
fn child(mut process: Child, mut each: impl FnMut(usize, f64, bool)) -> Result<Option<(usize, String)>> {
    let Some(out) = process.stdout.take() else {
        let _ = process.kill();
        process.wait().context("warm worker stdout unavailable, failed to reap worker")?;
        anyhow::bail!("warm worker stdout unavailable");
    };
    let process = Mutex::new(process);
    let (heard, over) = (Mutex::new(Instant::now()), AtomicBool::new(false));
    let hung = AtomicBool::new(false);
    let mut made = 0;
    std::thread::scope(|s| {
        s.spawn(|| {
            while !over.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if lock(&heard).elapsed() > LONGEST {
                    hung.store(true, Ordering::Relaxed);
                    let _ = lock(&process).kill();
                    return;
                }
            }
        });
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            *lock(&heard) = Instant::now();
            let mut words = line.split(' ');
            if words.next() == Some("made")
                && let (Some(n), Some(ms), Some(there)) = (words.next(), words.next(), words.next())
                && let (Ok(n), Ok(ms)) = (n.parse(), ms.parse())
            {
                made += 1;
                each(n, ms, there == "1");
            }
        }
        over.store(true, Ordering::Relaxed);
    });
    let status = lock(&process).wait();
    let well = status.as_ref().is_ok_and(|s| s.success());
    if well {
        return Ok(None);
    }
    let why = if hung.load(Ordering::Relaxed) {
        format!("hung for {} s", LONGEST.as_secs())
    } else {
        status.map_or_else(|e| e.to_string(), |s| format!("{s}"))
    };
    Ok(Some((made, why)))
}

/// What such a process does.
fn make(dir: &Path, these: &str) -> Result<()> {
    let these: Vec<usize> = these.split(',').map(|n| n.parse().context("--these")).collect::<Result<_>>()?;
    let cfg = victory_core::Config::load()?;
    victory_hle::known_pipelines::use_translations(cfg.cache_subdir("shaders")?);
    let mut out = std::io::stdout();
    known::make(dir, &these, |n, came| {
        let told = match came {
            None => writeln!(out, "at {n}"),
            Some((ms, there)) => writeln!(out, "made {n} {ms:.2} {}", u8::from(there)),
        };
        if told.and_then(|()| out.flush()).is_err() {
            // Whoever asked is gone.
            std::process::exit(1);
        }
    });
    // Without what a run's end does (nothing to save, the host's threads).
    let _ = out.flush();
    std::process::exit(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::process::CommandExt;

    fn fake_child(script: &str, stdout: Stdio) -> Result<Child> {
        let cmd = PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join(r"System32\cmd.exe");
        Ok(Command::new(cmd)
            .args(["/d", "/c"])
            .raw_arg(script)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(Stdio::null())
            .spawn()?)
    }

    #[test]
    fn used_refusals_are_retried_until_a_host_pipeline_is_made() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let dir = tmp.path();
        let entries: Vec<u8> = [0x11, 0x22, 0x33].into_iter()
            .flat_map(|byte| 16u32.to_le_bytes().into_iter().chain([byte; 16]))
            .collect();
        std::fs::write(dir.join("known-pipelines.bin"), entries)?;
        let keys = known::keys(dir);
        std::fs::write(dir.join("early-pipelines.txt"), keys[..2].join("\n"))?;
        std::fs::write(dir.join("warm-done.txt"), format!("fp {}\nother made {}\n", keys[0], keys[1]))?;

        // Both keys refuse at first. A second refusal of key 0 must still
        // give key 1 its turn, and once key 1 succeeds only key 0 remains.
        for (key, made) in [(0, 0), (1, 0), (0, 0), (1, 1), (0, 1)] {
            warm(dir, "fp", Some(1), Some(1), false, true, |batch| {
                assert_eq!(batch, [key]);
                fake_child(&format!("echo made {key} 1 {made}"), Stdio::piped())
            })?;
        }
        let records = std::fs::read_to_string(dir.join("warm-done.txt"))?;
        assert_eq!(records.lines().filter(|l| l.starts_with("fp made ")).count(), 2);
        warm(dir, "fp", Some(1), None, false, true, |_| panic!("both used pipelines were made"))
    }

    #[test]
    fn missing_executable_leaves_the_batch_pending() {
        let missing = std::env::temp_dir().join(format!("victory-missing-worker-{}.exe", std::process::id()));
        assert!(!missing.try_exists().unwrap());
        let starts = AtomicUsize::new(0);
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        let result = run_batches(
            &[7, 8, 9],
            1,
            |_| {
                starts.fetch_add(1, Ordering::Relaxed);
                Ok(Command::new(&missing).spawn()?)
            },
            |n, _, _| lock(&done).push(n),
            |n, _| lock(&bad).push(n),
        );
        assert_eq!(starts.load(Ordering::Relaxed), 1);
        assert!(lock(&done).is_empty());
        assert!(lock(&bad).is_empty());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("start warm worker"));
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn missing_stdout_is_a_startup_failure() {
        let error = run_batches(
            &[1, 2],
            1,
            |_| fake_child("exit /b 0", Stdio::null()),
            |_, _, _| panic!("unexpected completion"),
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("stdout"));
    }

    #[test]
    fn worker_exit_records_the_failed_pipeline_and_resumes() {
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        run_batches(
            &[7, 8, 9],
            1,
            |rest| match rest {
                [7, 8, 9] => fake_child("echo made 7 3.5 1&exit /b 7", Stdio::piped()),
                [9] => fake_child("echo made 9 2.0 0", Stdio::piped()),
                _ => panic!("unexpected batch {rest:?}"),
            },
            |n, ms, there| lock(&done).push((n, ms, there)),
            |n, why| lock(&bad).push((n, why)),
        )
        .unwrap();
        assert_eq!(*lock(&done), [(7, 3.5, true), (9, 2.0, false)]);
        let bad = lock(&bad);
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].0, 8);
        assert!(!bad[0].1.is_empty());
    }

    #[test]
    fn startup_failure_preserves_another_workers_completion() {
        // Hold the failed start until its peer has reported the only key in its chunk.
        let keys: Vec<_> = (0..=AT_ONCE).collect();
        let (confirmed, received) = std::sync::mpsc::channel();
        let received = Mutex::new(received);
        let error = run_batches(
            &keys,
            2,
            |chunk| {
                if chunk[0] == 0 {
                    assert_eq!(lock(&received).recv_timeout(Duration::from_secs(5)).unwrap(), AT_ONCE);
                    anyhow::bail!("worker unavailable")
                }
                fake_child(&format!("echo made {AT_ONCE} 1 1"), Stdio::piped())
            },
            |n, _, _| confirmed.send(n).unwrap(),
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("worker unavailable"));
    }
}
