#!/usr/bin/env node
"use strict";
/**
 * Minimal Vast.ai client for this image.
 *
 * Why Node: the DSH Windows sandbox breaks PowerShell/curl TLS paths, while
 * Node's OpenSSL works. No dependencies.
 *
 * Auth: VAST_API_KEY, or ~/.config/vastai/vast_api_key, ~/.vastai/vastai_api_key.
 *
 *   node tools/vast.js whoami
 *   node tools/vast.js instances
 *   node tools/vast.js search "A100 SXM4,H100 PCIE" 79 150
 *   node tools/vast.js rent --offer 123456 --image ghcr.io/...:latest --disk 150 \
 *        --label wan22-s2v --env WAN_API_KEY=... --port 8000:8000 --yes
 *   node tools/vast.js ssh 123456
 *   node tools/vast.js destroy 123456 --yes
 *
 * `rent` spends money. It refuses without --yes.
 */
const https = require("https");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const HOST = "console.vast.ai";
const IMAGE_DEFAULT = "ghcr.io/josephatwoosh/wan22-s2v-docker:latest";

function findKey() {
  if (process.env.VAST_API_KEY) return { key: process.env.VAST_API_KEY.trim(), from: "env VAST_API_KEY" };
  const candidates = [
    path.join(os.homedir(), ".config", "vastai", "vast_api_key"),
    path.join(os.homedir(), ".vastai", "vastai_api_key"),
    path.join(process.env.APPDATA || "", "vastai", "vast_api_key"),
  ];
  for (const candidate of candidates) {
    try {
      const key = fs.readFileSync(candidate, "utf8").trim();
      if (key) return { key, from: candidate };
    } catch {
      /* next */
    }
  }
  return null;
}

function request(method, apiPath, body, key) {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? null : JSON.stringify(body);
    const headers = { Accept: "application/json" };
    if (key) headers.Authorization = `Bearer ${key}`;
    if (data) {
      headers["Content-Type"] = "application/json";
      headers["Content-Length"] = Buffer.byteLength(data);
    }
    const req = https.request({ hostname: HOST, path: apiPath, method, headers, timeout: 45000 }, (res) => {
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
    });
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error("request timed out")));
    if (data) req.write(data);
    req.end();
  });
}

function needKey() {
  const found = findKey();
  if (!found) {
    console.error(
      "No Vast API key. Set VAST_API_KEY, run `vastai set api-key <key>`, " +
        "or write the key to ~/.config/vastai/vast_api_key",
    );
    process.exit(2);
  }
  return found;
}

function parseFlags(argv) {
  const flags = { env: [], _: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--yes") flags.yes = true;
    else if (arg === "--allow-small") flags.allowSmall = true;
    else if (arg === "--allow-inet") flags.allowInet = true;
    else if (arg === "--port") flags.env.push(`-p ${argv[++i]}`);
    else if (arg === "--env") flags.env.push(argv[++i]);
    else if (arg === "--offer") flags.offer = argv[++i];
    else if (arg === "--image") flags.image = argv[++i];
    else if (arg === "--disk") flags.disk = Number(argv[++i]);
    else if (arg === "--label") flags.label = argv[++i];
    else if (arg === "--onstart") flags.onstart = argv[++i];
    else if (arg === "--name") flags.name = argv[++i];
    else if (arg === "--max-inet") flags.maxInet = Number(argv[++i]);
    else if (arg === "--ram-gb") flags.ramGb = Number(argv[++i]);
    else if (arg === "--driver") flags.driver = Number(argv[++i]);
    else if (arg === "--transfer-gb") flags.transferGb = Number(argv[++i]);
    else flags._.push(arg);
  }
  return flags;
}

function envMap(list) {
  const env = {};
  for (const item of list) {
    if (item.startsWith("-p ")) {
      const [key, value] = item.slice(3).split("=");
      env[`-p ${key}`] = value || "1";
      continue;
    }
    const index = item.indexOf("=");
    if (index < 1) continue;
    env[item.slice(0, index)] = item.slice(index + 1);
  }
  return env;
}

/** Vast returns both a $/GB field and an explicit $/TB field; prefer the latter. */
function perTb(offer, tbKey, gbKey) {
  if (typeof offer[tbKey] === "number") return offer[tbKey];
  return (offer[gbKey] || 0) * 1024;
}

/**
 * Filter VRAM locally on `gpu_total_ram` (MB).
 *
 * Vast's bundle filter and its offer field disagree on units (the same lesson
 * that produced "gpu_ram>=23" matching nothing), so the search asks for a loose
 * pool and this file does the comparison in MB, which is what the offer carries.
 */
function vramGb(offer) {
  return Math.round((offer.gpu_total_ram || 0) / 1024);
}

function ramGb(offer) {
  return Math.round((offer.cpu_ram || 0) / 1024);
}

/** "580.95.05" -> 580, so the CUDA 12.8 driver floor can be compared. */
function driverMajor(offer) {
  const major = Number(String(offer.driver_version || "0").split(".")[0]);
  return Number.isFinite(major) ? major : 0;
}

async function offers(names, minVramGb, minDiskGb, key) {
  const query = {
    gpu_name: { in: names },
    num_gpus: { gte: 1 },
    disk_space: { gte: minDiskGb },
    rentable: { eq: true },
    type: "ondemand",
    order: [["dph_total", "asc"]],
    limit: 1000,
  };
  const r = await request("POST", "/api/v0/bundles/", query, key);
  if (r.status !== 200) return { status: r.status, list: [], body: r.body };
  const list = (r.body.offers || []).filter((o) => vramGb(o) >= minVramGb);
  return { status: r.status, list, body: r.body };
}

async function offerById(id, key) {
  // The bundle filter has no id predicate; fetch the pool the search uses and
  // match locally.
  const pool = await offers(["A100 SXM4", "A100 PCIE", "H100 PCIE", "H100 SXM", "L40S", "A6000"], 40, 60, key);
  return pool.list.find((o) => String(o.id) === String(id)) || null;
}

function sshTarget(instance) {
  const host = instance.ssh_host || instance.public_ipaddr;
  const port = instance.ssh_port || 22;
  return { host, port };
}

async function instanceById(id, key) {
  const r = await request("GET", "/api/v1/instances/", undefined, key.key);
  const list = r.body.instances || r.body;
  return (Array.isArray(list) ? list : []).find((i) => String(i.id) === String(id)) || null;
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const flags = parseFlags(rest);

  switch (cmd) {
    case "whoami": {
      const key = needKey();
      const r = await request("GET", "/api/v0/users/current/", undefined, key.key);
      console.log("key source:", key.from);
      console.log("status:", r.status);
      console.log(
        JSON.stringify({ id: r.body.id, email: r.body.email, credit: r.body.credit }, null, 2),
      );
      break;
    }

    case "instances": {
      const key = needKey();
      const r = await request("GET", "/api/v1/instances/", undefined, key.key);
      const list = r.body.instances || r.body;
      if (!Array.isArray(list)) {
        console.log(JSON.stringify(r.body, null, 2).slice(0, 4000));
        break;
      }
      console.log(`status ${r.status}, ${list.length} instance(s)`);
      console.table(
        list.map((i) => ({
          id: i.id,
          gpu: `${i.num_gpus || 1}x ${i.gpu_name}`,
          status: i.actual_status || i.status,
          ssh: `${i.ssh_host || i.public_ipaddr || "?"}:${i.ssh_port || 22}`,
          dph: i.dph_total,
          label: i.label,
        })),
      );
      break;
    }

    case "search": {
      const names = (flags._[0] || "A100 SXM4,A100 PCIE,H100 PCIE,H100 SXM").split(",").map((s) => s.trim());
      const minVram = Number(flags._[1] || 79);
      const minDisk = Number(flags._[2] || 150);
      const minRam = flags.ramGb ?? 64;
      const minDriver = flags.driver ?? 570;
      const maxInet = flags.maxInet ?? 4;
      const key = findKey();
      const pool = await offers(names, minVram, minDisk, key?.key);
      if (pool.status !== 200) {
        console.error("status", pool.status, JSON.stringify(pool.body).slice(0, 500));
        process.exit(1);
      }
      const ranked = pool.list
        .map((o) => ({
          ...o,
          downTb: perTb(o, "internet_down_cost_per_tb", "inet_down_cost"),
          upTb: perTb(o, "internet_up_cost_per_tb", "inet_up_cost"),
        }))
        .filter(
          (o) =>
            ramGb(o) >= minRam &&
            driverMajor(o) >= minDriver &&
            o.downTb < maxInet &&
            o.upTb < maxInet,
        );
      if (!ranked.length) {
        console.error(
          `No offers with >= ${minVram} GB VRAM, >= ${minRam} GB RAM, >= ${minDisk} GB disk, ` +
            `driver >= ${minDriver} and < $${maxInet}/TB both ways.`,
        );
        process.exit(1);
      }
      console.table(
        ranked.slice(0, 15).map((o) => ({
          offer: o.id,
          machine: o.machine_id,
          gpu: `${o.num_gpus}x ${o.gpu_name}`,
          vram_gb: vramGb(o),
          ram_gb: ramGb(o),
          disk_gb: Math.round(o.disk_space || 0),
          down_tb: Number(o.downTb.toFixed(2)),
          up_tb: Number(o.upTb.toFixed(2)),
          weights_usd: Number(((45.6 / 1024) * o.downTb).toFixed(2)),
          mbps: `${Math.round(o.inet_down || 0)}/${Math.round(o.inet_up || 0)}`,
          driver: o.driver_version,
          rel: Number((o.reliability2 ?? o.reliability ?? 0).toFixed(2)),
          geo: String(o.geolocation || "").split(",").pop().trim(),
          usd_hr: Number((o.dph_total || 0).toFixed(3)),
        })),
      );
      console.log(
        `\n${ranked.length} offer(s): >= ${minVram} GB VRAM, >= ${minRam} GB RAM, ` +
          `>= ${minDisk} GB disk, driver >= ${minDriver}, < $${maxInet}/TB each way.`,
      );
      console.log(
        "Total for the first run: 45.6 GB down (weights) + ~8 GB image + 1 GB up, plus instance time.",
      );
      break;
    }

    case "rent": {
      const key = needKey();
      if (!flags.offer) {
        console.error("rent needs --offer <id> (see: search)");
        process.exit(2);
      }
      if (!flags.yes) {
        console.error("Refusing to spend money without --yes");
        process.exit(2);
      }
      const offer = await offerById(flags.offer, key.key);
      if (offer && !flags.allowSmall && vramGb(offer) < 48) {
        console.error(
          `Refusing: offer ${flags.offer} has ${vramGb(offer)} GB VRAM; the DiT shards are 32.6 GB ` +
            "in bf16 and need at least 48 GB (80 GB recommended). Override with --allow-small if " +
            "you know better.",
        );
        process.exit(2);
      }
      if (offer && !flags.allowSmall && ramGb(offer) < 64) {
        console.error(
          `Refusing: offer ${flags.offer} has ${ramGb(offer)} GB host RAM; the offloaded DiT ` +
            "(32.6 GB) and the text encoder (11.4 GB) are resident at once, so 64 GB is the " +
            "minimum. Override with --allow-small if you know better.",
        );
        process.exit(2);
      }
      if (offer && driverMajor(offer) < 570) {
        console.error(
          `Refusing: offer ${flags.offer} runs driver ${offer.driver_version}; CUDA 12.8 needs 570+.`,
        );
        process.exit(2);
      }
      if (offer) {
        const downTb = perTb(offer, "internet_down_cost_per_tb", "inet_down_cost");
        const upTb = perTb(offer, "internet_up_cost_per_tb", "inet_up_cost");
        const maxInet = flags.maxInet ?? 4;
        const transferGb = flags.transferGb ?? 50;
        console.log(
          `offer ${flags.offer}: ${vramGb(offer)} GB VRAM, ${ramGb(offer)} GB RAM, ` +
            `$${Number(offer.dph_total).toFixed(3)}/h, ` +
            `down $${downTb.toFixed(2)}/TB up $${upTb.toFixed(2)}/TB`,
        );
        console.log(
          `projected transfer (${transferGb} GB down + 1 GB up): ` +
            `$${(((transferGb / 1024) * downTb) + ((1 / 1024) * upTb)).toFixed(2)}`,
        );
        if (!flags.allowInet && (downTb >= maxInet || upTb >= maxInet)) {
          console.error(
            `Refusing: transfer is not under $${maxInet}/TB both ways ` +
              `(down $${downTb.toFixed(2)}, up $${upTb.toFixed(2)}). Use --allow-inet to override.`,
          );
          process.exit(2);
        }
      }
      const body = {
        client_id: "me",
        image: flags.image || IMAGE_DEFAULT,
        disk: flags.disk || 150,
        runtype: "ssh",
        label: flags.label || "wan22-s2v",
        env: envMap(flags.env),
      };
      if (flags.onstart) body.onstart = flags.onstart;
      const r = await request("PUT", `/api/v0/asks/${flags.offer}/`, body, key.key);
      console.log("status:", r.status);
      console.log(JSON.stringify(r.body, null, 2).slice(0, 1200));
      if (r.status >= 300) process.exit(1);
      console.log("\nNext: node tools/vast.js instances   (wait for 'running')");
      console.log("Then: ssh in, `wan22-s2v download`, `wan22-s2v doctor`.");
      break;
    }

    case "ssh": {
      const key = needKey();
      const inst = await instanceById(flags._[0], key);
      if (!inst) {
        console.error(`no instance ${flags._[0]}`);
        process.exit(1);
      }
      const { host, port } = sshTarget(inst);
      console.log(`ssh -p ${port} root@${host}`);
      const result = spawnSync("ssh", ["-p", String(port), `root@${host}`], {
        stdio: ["inherit", "inherit", "inherit"],
      });
      process.exit(result.status === null ? 1 : result.status);
      break;
    }

    case "stop":
    case "start": {
      const key = needKey();
      const id = flags._[0];
      if (!id) {
        console.error(`${cmd} <instance_id>`);
        process.exit(2);
      }
      const state = cmd === "stop" ? "stopped" : "running";
      const r = await request("PUT", `/api/v0/instances/${id}/`, { state }, key.key);
      console.log(`${cmd} ${id} -> ${r.status} ${JSON.stringify(r.body).slice(0, 300)}`);
      break;
    }

    case "destroy": {
      const key = needKey();
      const id = flags._[0];
      if (!id || !flags.yes) {
        console.error("Refusing to destroy without an instance id and --yes");
        process.exit(2);
      }
      const r = await request("DELETE", `/api/v0/instances/${id}/`, undefined, key.key);
      console.log("status:", r.status, JSON.stringify(r.body).slice(0, 500));
      break;
    }

    default:
      console.error(
        "usage: node tools/vast.js {whoami|instances|search|rent|ssh|stop|start|destroy} [...]\n" +
          "  search 'A100 SXM4,H100 PCIE' 79 150 [--ram-gb 64] [--driver 570] [--max-inet 4]\n" +
          "  rent --offer 123 --disk 150 --env WAN_API_KEY=... --port 8000:8000 --yes",
      );
      process.exit(2);
  }
}

main().catch((err) => {
  console.error("error:", err.message);
  process.exit(1);
});
