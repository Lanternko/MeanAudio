import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { chromium } from "playwright";

const OUT = "/tmp/three-mainlines-pdf";
const PAGES = 7;
const BASE = "http://127.0.0.1:5174";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });

await page.goto(BASE + "/s/three-mainlines", { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(2000);

// Enter present / fullscreen
const present = page.getByRole("button", { name: /Present/i }).first();
if (await present.count()) {
  await present.click();
  await page.waitForTimeout(1000);
}
await page.keyboard.press("f");
await page.waitForTimeout(800);
await page.keyboard.press("Home");
await page.waitForTimeout(500);

console.log("URL", page.url());
const t0 = await page.locator("body").innerText();
console.log("T0", t0.replace(/\s+/g, " ").slice(0, 200));

for (let i = 0; i < PAGES; i++) {
  if (i > 0) {
    await page.keyboard.press("ArrowRight");
    await page.waitForTimeout(800);
  }
  // try to screenshot only the slide stage
  const path = join(OUT, `page-${String(i + 1).padStart(2, "0")}.png`);
  const stage = page.locator('[class*="canvas"], [data-canvas], .osd-canvas, main').first();
  if (await stage.count()) {
    await stage.screenshot({ path, type: "png" });
  } else {
    await page.screenshot({ path, type: "png", fullPage: false });
  }
  const t = await page.locator("body").innerText().catch(() => "");
  console.log("shot", i + 1, t.replace(/\s+/g, " ").slice(0, 120));
}
await browser.close();

writeFileSync("/tmp/pngs_to_pdf.py", `
from pathlib import Path
from PIL import Image
out = Path("/tmp/three-mainlines-phase8.pdf")
imgs = [Image.open(p).convert("RGB") for p in sorted(Path("${OUT}").glob("page-*.png"))]
print("n", len(imgs), [im.size for im in imgs])
# normalize size to 1920x1080
norm=[]
for im in imgs:
    if im.size != (1920,1080):
        im = im.resize((1920,1080), Image.Resampling.LANCZOS)
    norm.append(im)
norm[0].save(out, save_all=True, append_images=norm[1:])
print("PDF", out, out.stat().st_size)
`);
execSync("python3 /tmp/pngs_to_pdf.py", { stdio: "inherit" });
