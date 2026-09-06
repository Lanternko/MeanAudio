import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
// resolve playwright from open-slide node_modules
process.chdir("/home/kojiek/MeanAudio/open-slide");

const { chromium: cr } = await import("/home/kojiek/MeanAudio/open-slide/node_modules/playwright/index.mjs").catch(async () => {
  return await import("playwright");
});

const OUT = "/tmp/quarter-scale-pdf";
const SLIDE = "quarter-scale";
const PAGES = 6;
const BASE = "http://127.0.0.1:5174";

mkdirSync(OUT, { recursive: true });

const browser = await cr.launch({ headless: true });
const page = await browser.newPage({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 1,
});

await page.goto(BASE + "/", { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(1500);
const homeText = await page.locator("body").innerText();
console.log("HOME_HEAD", homeText.replace(/\s+/g, " ").slice(0, 400));

// Click the quarter-scale card/link
const selectors = [
  "text=Quarter-scale",
  "text=K-ablation",
  "text=Caption10s fix",
  "a:has-text('quarter')",
];
let opened = false;
for (const sel of selectors) {
  const loc = page.locator(sel).first();
  if ((await loc.count()) > 0) {
    console.log("click", sel);
    await loc.click();
    await page.waitForTimeout(2000);
    opened = true;
    break;
  }
}
if (!opened) {
  // direct navigate variants
  for (const u of [`${BASE}/slides/${SLIDE}`, `${BASE}/${SLIDE}`, `${BASE}/play/${SLIDE}`]) {
    await page.goto(u, { waitUntil: "networkidle" });
    await page.waitForTimeout(1000);
    const t = await page.locator("body").innerText();
    console.log("nav", u, t.replace(/\s+/g, " ").slice(0, 150));
    if (/PHASE 8|Quarter-scale|QUARTER SCALE|caption/i.test(t)) {
      opened = true;
      break;
    }
  }
}

console.log("URL", page.url());
// Enter presenter if button exists
for (const label of [/Play/i, /Present/i, /Fullscreen/i, /開始/]) {
  const b = page.getByRole("button", { name: label }).first();
  if (await b.count()) {
    await b.click().catch(() => {});
    await page.waitForTimeout(500);
  }
}
await page.keyboard.press("f").catch(() => {});
await page.waitForTimeout(400);

const pngs = [];
for (let i = 0; i < PAGES; i++) {
  if (i > 0) {
    await page.keyboard.press("ArrowRight");
    await page.waitForTimeout(700);
  } else {
    await page.waitForTimeout(800);
  }
  const path = join(OUT, `page-${String(i + 1).padStart(2, "0")}.png`);
  // Prefer largest visible "slide" frame
  const frames = page.locator("iframe");
  if ((await frames.count()) > 0) {
    const frame = frames.first();
    await frame.screenshot({ path, type: "png" });
  } else {
    await page.screenshot({ path, type: "png", fullPage: false });
  }
  pngs.push(path);
  const t = await page.locator("body").innerText().catch(() => "");
  console.log("shot", path, "text=", t.replace(/\s+/g, " ").slice(0, 100));
}

await browser.close();

writeFileSync(
  "/tmp/pngs_to_pdf.py",
  `
from pathlib import Path
from PIL import Image
out = Path("/tmp/quarter-scale-open-slide.pdf")
imgs = []
for p in sorted(Path("${OUT}").glob("page-*.png")):
    im = Image.open(p).convert("RGB")
    imgs.append(im)
    print(p, im.size)
if not imgs:
    raise SystemExit("no images")
imgs[0].save(out, save_all=True, append_images=imgs[1:])
print("PDF", out, out.stat().st_size)
`
);
execSync("python3 /tmp/pngs_to_pdf.py", { stdio: "inherit" });
