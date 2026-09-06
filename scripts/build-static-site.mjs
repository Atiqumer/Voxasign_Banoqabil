import { cp, mkdir, rm, stat } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = join(root, "deploy");

// Keep this list explicit: anything absent is development-only and is never
// published by Netlify (training data, checkpoints, reports, and local paths).
const runtimeFiles = [
  "index.html",
  "studio.html",
  "script.js",
  "psl_v2_features.js",
  "demo.mp4",
  "img.png",
  "web_model/model.json",
  "web_model/weights.json",
  "web_model/psl_v2/model.json",
  "web_model/psl_v2/weights.json",
  "web_model/psl_v2/scaler.json",
  "web_model/psl_v2/class_map.json",
];

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

for (const file of runtimeFiles) {
  const source = join(root, file);
  const destination = join(output, file);
  await stat(source);
  await mkdir(dirname(destination), { recursive: true });
  await cp(source, destination);
}

console.log(`Prepared ${runtimeFiles.length} runtime files in ${relative(root, output)}.`);
