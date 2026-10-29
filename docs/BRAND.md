# AquaSOT Brand

The visual identity of AquaSOT: mark, wordmark, palette, typefaces, and
usage rules. Asset files live in `assets/brand/`.

## Concept

The mark draws the geometry the project is about: a wide forward-looking
sonar fan (FLS, 120 to 160 deg in practice) encloses the narrow RGB camera
cone (60 to 70 deg), and the target bracket sits where the two overlap.

## Mark ("Twin Fan")

Geometry in a 64 x 64 unit box:

| Element | Definition |
|---|---|
| Sensor origin (apex) | (32, 49) |
| Sonar fan | 130 deg wedge, radius 34, fill at 20% opacity, solid rim stroke 2.2 |
| Range rings | radius 34/3 and 2*34/3, stroke 1.4 at 75% opacity |
| RGB cone | 65 deg wedge, radius 34, solid fill |
| Target bracket | 11 x 11 corner bracket centered at (32, 27), stroke 2.4, cut out of the cone (transparent) |

Simplified mark (below 24 px, favicon): no range rings, solid fan, and the
bracket becomes a solid 10 x 10 square cut-out.

## Wordmark ("W1")

`Aqua` in Sora SemiBold (600), tracking -0.02 em; `SOT` in IBM Plex Mono
SemiBold (600), colored with the RGB color, 0.04 em after `Aqua`. Keep the
mixed case: it matches the repository and package name.

Horizontal lockup: the mark box is 1.5 x the wordmark font size, the gap is
0.42 em, and the mark's visual center aligns with the wordmark's cap-height
center. Stacked lockup: the mark box is 2.5 x the font size, centered above
the wordmark with a 0.4 em gap.

## Palette

Cyan always means RGB; amber always means sonar. Apply the same mapping in
paper figures and plots, so either modality reads the same everywhere.

| Token | Hex | Role |
|---|---|---|
| Abyss | `#06141F` | Dark background |
| Trench | `#0C2231` | Dark panel |
| Foam | `#E8F4F5` | Text on dark |
| Ink | `#0B1D2A` | Text on light |
| Biolume | `#3FE0D0` | RGB, on dark |
| Tide | `#0A7C86` | RGB, on light |
| Ping | `#FFB547` | Sonar, on dark |
| Ping deep | `#C77700` | Sonar graphics on light |
| Ping text | `#9A5B00` | Sonar text on light |

Contrast (WCAG): Biolume on Abyss 11.3:1, Ping on Abyss 10.6:1, Foam on
Abyss 16.6:1, Tide on white 5.0:1, Ping text on white 5.4:1. Ping deep on
white is 3.5:1: use it for graphics and large text only.

## Typefaces

| Role | Family |
|---|---|
| Display and wordmark | Sora |
| Body | IBM Plex Sans |
| Data, code, metrics | IBM Plex Mono |

All three are under the SIL Open Font License 1.1. The SVG assets contain
outlined glyphs only, so they render without the fonts installed; the font
files are not part of this repository.

## Usage rules

- Clear space around the mark: at least one quarter of the mark width.
- Minimum size: 24 px for the full mark; below that, use the simplified mark.
- Do not stretch, rotate, recolor, or swap the two modality colors.
- Do not place the mark on busy images or gradients; use Abyss, white, or a
  plain theme background.

## Files

The brand kit lives in `assets/brand/` and ships with source archives.
The README header images in `.github/assets/` are copies of the horizontal
lockups kept there by the repository docs convention; `.github/` is
excluded from `git archive`, so they exist only on the forge. Change both
copies together.

| File | Size | Use |
|---|---|---|
| `assets/brand/mark-light.svg` | 64 x 64 | Mark on light backgrounds |
| `assets/brand/mark-dark.svg` | 64 x 64 | Mark on dark backgrounds |
| `assets/brand/mark-simple-light.svg` | 64 x 64 | Simplified mark on light backgrounds, below 24 px |
| `assets/brand/mark-simple-dark.svg` | 64 x 64 | Simplified mark on dark backgrounds, below 24 px |
| `assets/brand/wordmark-light.svg` | 379 x 98 | Wordmark alone, light backgrounds (footers, slides) |
| `assets/brand/wordmark-dark.svg` | 379 x 98 | Wordmark alone, dark backgrounds |
| `assets/brand/lockup-horizontal-light.svg` | 536 x 114 | Mark beside wordmark, light backgrounds |
| `assets/brand/lockup-horizontal-dark.svg` | 536 x 114 | Mark beside wordmark, dark backgrounds |
| `assets/brand/lockup-stacked-light.svg` | 308 x 197 | Mark above wordmark, light backgrounds |
| `assets/brand/lockup-stacked-dark.svg` | 308 x 197 | Mark above wordmark, dark backgrounds |
| `assets/brand/favicon.svg` | 64 x 64 | Simplified mark; follows the browser color scheme |
| `assets/brand/favicon-16.png`, `favicon-32.png`, `favicon.ico` | 16, 32, 48 | Raster favicon: simplified mark on a rounded Abyss tile, for hosts without SVG icons |
| `assets/brand/avatar.svg`, `avatar.png` | 512 x 512 | Organization avatar (for example Hugging Face) |
| `assets/brand/social-preview.svg`, `social-preview.png` | 1280 x 640 | GitHub social preview (Settings, General, Social preview) |
| `.github/assets/banner-light.svg`, `banner-dark.svg` | 536 x 114 | README header (copies of the horizontal lockups) |
