#!/usr/bin/env node
"use strict";
/**
 * Publish a directory to GitHub without git, as ONE commit.
 *
 * Uses the Git Data API (blobs -> tree -> commit -> ref) instead of the
 * Contents API, which has no multi-file mode. One commit also means one
 * Actions run per publish instead of one per file.
 *
 * File modes are set from the path: shell scripts and CLI JS are 100755, the
 * rest 100644. Windows has no executable bit, so this is the only way a clone
 * gets `./scripts/run.sh` working.
 *
 * Usage:
 *   node tools/publish.js --dir . --repo JosephAtwoosh/wan22-s2v-docker \
 *        --message "Initial commit: Wan2.2-S2V renderer" \
 *        [--dry-run] [--prune] [--description "..."]
 *
 * Token: GITHUB_TOKEN / GH_TOKEN, or --token-file (default C:\dsh\pat.txt).
 * The token is never printed.
 */
const crypto = require("crypto");
const fs = require("fs");
const https = require("https");
const path = require("path");

const OWNER_DEFAULT = "JosephAtwoosh";
const REPO_DEFAULT = "wan22-s2v-docker";
const SKIP = new Set([".git", "__pycache__", ".pytest_cache", "node_modules", "models", "output", "performances"]);

function parseArgs(argv) {
  const args = {
    dir: ".",
    repo: `${OWNER_DEFAULT}/${REPO_DEFAULT}`,
    branch: "main",
    message: "Publish",
    tokenFile: "C:\\dsh\\pat.txt",
    dryRun: false,
    prune: false,
    description: null,
    tag: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    const value = () => argv[++i];
    if (flag === "--dir") args.dir = value();
    else if (flag === "--repo") args.repo = value();
    else if (flag === "--branch") args.branch = value();
    else if (flag === "--message") args.message = value();
    else if (flag === "--token-file") args.tokenFile = value();
    else if (flag === "--description") args.description = value();
    else if (flag === "--tag") args.tag = value();
    else if (flag === "--dry-run") args.dryRun = true;
    else if (flag === "--prune") args.prune = true;
    else {
      console.error(`unknown flag ${flag}`);
      process.exit(2);
    }
  }
  return args;
}

function token(args) {
  const fromEnv = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
  if (fromEnv) return fromEnv.trim();
  try {
    const value = fs.readFileSync(args.tokenFile, "utf8").trim();
    if (value) return value;
  } catch {
    /* fall through */
  }
  console.error("No token: set GITHUB_TOKEN or pass --token-file");
  process.exit(2);
}

function api(auth, method, apiPath, body) {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? null : JSON.stringify(body);
    const headers = {
      Authorization: `Bearer ${auth}`,
      "User-Agent": "wan22-s2v-publish",
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
    };
    if (data) {
      headers["Content-Type"] = "application/json";
      headers["Content-Length"] = Buffer.byteLength(data);
    }
    const req = https.request(
      { hostname: "api.github.com", path: apiPath, method, headers, timeout: 60000 },
      (res) => {
        let buffer = "";
        res.on("data", (chunk) => (buffer += chunk));
        res.on("end", () => {
          let parsed = buffer;
          try {
            parsed = JSON.parse(buffer);
          } catch {
            /* text */
          }
          resolve({ status: res.statusCode, body: parsed });
        });
      },
    );
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error("request timed out")));
    if (data) req.write(data);
    req.end();
  });
}

function walk(root, prefix = "", files = []) {
  const entries = fs.readdirSync(path.join(root, prefix), { withFileTypes: true });
  for (const entry of entries) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      if (SKIP.has(entry.name)) continue;
      walk(root, relative, files);
    } else if (entry.isFile()) {
      if (entry.name === ".DS_Store") continue;
      files.push(relative);
    }
  }
  return files;
}

function blobSha(buffer) {
  const header = Buffer.from(`blob ${buffer.length}\0`, "utf8");
  return crypto.createHash("sha1").update(Buffer.concat([header, buffer])).digest("hex");
}

function modeFor(relative) {
  if (/\.sh$/.test(relative)) return "100755";
  if (/^(tools|examples)\/.*\.js$/.test(relative)) return "100755";
  return "100644";
}

function textCheck(buffer, relative) {
  const textExtensions = /\.(py|sh|md|txt|json|ya?ml|toml|js|css|html|example|gitignore|gitattributes|dockerignore)$/;
  const name = path.basename(relative);
  const looksText =
    textExtensions.test(relative) || name.startsWith(".") || name === "Dockerfile" || name === "LICENSE";
  if (!looksText) return;
  if (buffer.toString("utf8").includes("\r\n")) {
    throw new Error(`${relative} has CRLF line endings; convert to LF before publishing`);
  }
}

async function readHead(auth, owner, repo, branch) {
  const ref = await api(auth, "GET", `/repos/${owner}/${repo}/git/ref/heads/${branch}`);
  if (ref.status === 404 || ref.status === 409) {
    // 409 is "Git Repository is empty".
    return { headSha: null, baseTree: null, remote: new Map(), empty: true };
  }
  if (ref.status !== 200) {
    console.error(`cannot read ref: HTTP ${ref.status} ${JSON.stringify(ref.body).slice(0, 300)}`);
    process.exit(1);
  }
  const headSha = ref.body.object.sha;
  const commit = await api(auth, "GET", `/repos/${owner}/${repo}/git/commits/${headSha}`);
  if (commit.status !== 200) {
    console.error(`cannot read head commit: HTTP ${commit.status}`);
    process.exit(1);
  }
  const baseTree = commit.body.tree.sha;
  const tree = await api(auth, "GET", `/repos/${owner}/${repo}/git/trees/${baseTree}?recursive=1`);
  const remote = new Map();
  for (const entry of (tree.body && tree.body.tree) || []) {
    if (entry.type === "blob") remote.set(entry.path, entry.sha);
  }
  return { headSha, baseTree, remote, empty: false };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const auth = token(args);
  const [owner, repo] = args.repo.split("/");
  const root = path.resolve(args.dir);

  const local = walk(root).map((relative) => {
    const buffer = fs.readFileSync(path.join(root, relative));
    textCheck(buffer, relative);
    return { relative, buffer, mode: modeFor(relative), sha: blobSha(buffer) };
  });
  console.log(`${local.length} local files under ${root}`);

  // Current head, if any.
  let head = await readHead(auth, owner, repo, args.branch);
  if (head.empty) {
    if (args.dryRun) {
      console.log(`dry run: repository is empty, would publish ${local.length} files`);
      return;
    }
    // The Git Data API refuses blobs in a repository with zero commits
    // ("Git Repository is empty"), so the first commit has to come from the
    // Contents API. One real file makes a valid seed; the tree commit then
    // adds everything else on top.
    const seed = local.find((file) => file.relative === "LICENSE") || local[0];
    console.log(`empty repository: seeding with ${seed.relative} via the Contents API`);
    const seeded = await api(auth, "PUT", `/repos/${owner}/${repo}/contents/${encodeURI(seed.relative)}`, {
      message: `${args.message}\n\nBootstrap commit: the Git Data API cannot write blobs into an empty repository.`,
      content: seed.buffer.toString("base64"),
      branch: args.branch,
    });
    if (seeded.status !== 200 && seeded.status !== 201) {
      console.error(`seed failed: HTTP ${seeded.status} ${JSON.stringify(seeded.body).slice(0, 300)}`);
      process.exit(1);
    }
    head = await readHead(auth, owner, repo, args.branch);
  }
  const { headSha, baseTree, remote } = head;
  console.log(`head ${headSha.slice(0, 12)} with ${remote.size} blob(s)`);

  const creates = local.filter((file) => !remote.has(file.relative));
  const updates = local.filter((file) => remote.has(file.relative) && remote.get(file.relative) !== file.sha);
  const unchanged = local.length - creates.length - updates.length;
  const removals = args.prune
    ? [...remote.keys()].filter((relative) => !local.some((file) => file.relative === relative))
    : [];

  console.log(
    `${creates.length} to create, ${updates.length} to update, ${unchanged} unchanged` +
      (args.prune ? `, ${removals.length} to delete` : " (--prune to delete extras)"),
  );
  for (const file of [...creates, ...updates]) {
    console.log(`  ${remote.has(file.relative) ? "update" : "create"} ${file.relative}`);
  }
  for (const relative of removals) console.log(`  delete ${relative}`);
  if (args.dryRun) {
    console.log("dry run: nothing written");
    return;
  }
  if (!creates.length && !updates.length && !removals.length) {
    console.log("nothing to publish");
    return;
  }

  const entries = [];
  for (const file of [...creates, ...updates]) {
    const blob = await api(auth, "POST", `/repos/${owner}/${repo}/git/blobs`, {
      content: file.buffer.toString("base64"),
      encoding: "base64",
    });
    if (blob.status !== 201) {
      console.error(`blob failed for ${file.relative}: HTTP ${blob.status} ${JSON.stringify(blob.body).slice(0, 300)}`);
      process.exit(1);
    }
    entries.push({ path: file.relative, mode: file.mode, type: "blob", sha: blob.body.sha });
  }
  for (const relative of removals) {
    entries.push({ path: relative, mode: "100644", type: "blob", sha: null });
  }

  const tree = await api(auth, "POST", `/repos/${owner}/${repo}/git/trees`, {
    ...(baseTree ? { base_tree: baseTree } : {}),
    tree: entries,
  });
  if (tree.status !== 201) {
    console.error(`tree failed: HTTP ${tree.status} ${JSON.stringify(tree.body).slice(0, 500)}`);
    process.exit(1);
  }

  const commit = await api(auth, "POST", `/repos/${owner}/${repo}/git/commits`, {
    message: args.message,
    tree: tree.body.sha,
    parents: headSha ? [headSha] : [],
  });
  if (commit.status !== 201) {
    console.error(`commit failed: HTTP ${commit.status} ${JSON.stringify(commit.body).slice(0, 500)}`);
    process.exit(1);
  }

  if (headSha) {
    const updated = await api(auth, "PATCH", `/repos/${owner}/${repo}/git/refs/heads/${args.branch}`, {
      sha: commit.body.sha,
      force: false,
    });
    if (updated.status !== 200) {
      console.error(`ref update failed: HTTP ${updated.status} ${JSON.stringify(updated.body).slice(0, 300)}`);
      process.exit(1);
    }
  } else {
    const created = await api(auth, "POST", `/repos/${owner}/${repo}/git/refs`, {
      ref: `refs/heads/${args.branch}`,
      sha: commit.body.sha,
    });
    if (created.status !== 201) {
      console.error(`ref create failed: HTTP ${created.status} ${JSON.stringify(created.body).slice(0, 300)}`);
      process.exit(1);
    }
  }

  if (args.description) {
    const patched = await api(auth, "PATCH", `/repos/${owner}/${repo}`, { description: args.description });
    console.log(`description -> HTTP ${patched.status}`);
  }

  if (args.tag) {
    const tagged = await api(auth, "POST", `/repos/${owner}/${repo}/git/refs`, {
      ref: `refs/tags/${args.tag}`,
      sha: commit.body.sha,
    });
    console.log(
      `tag ${args.tag} -> HTTP ${tagged.status}` +
        (tagged.status === 201 ? " (a build-* tag triggers the image workflow)" : ""),
    );
  }

  console.log(`\ncommitted ${commit.body.sha.slice(0, 12)} (${entries.length} tree entries)`);
  console.log(`https://github.com/${owner}/${repo}/commit/${commit.body.sha}`);
}

main().catch((error) => {
  console.error("error:", error.message);
  process.exit(1);
});
