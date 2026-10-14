// result.json을 zod 스키마(lib/types.ts)로 검증하고, 통과하면 palette.ts LEVEL_STYLES를 JSON으로 출력한다.
//   node scripts/validate-result.mjs <result.json>     (cv/tests/test_schema.py가 호출)
import { readFileSync } from "node:fs";
import { LEVEL_STYLES } from "../lib/palette.ts";
import { InspectionResult } from "../lib/types.ts";

const r = InspectionResult.safeParse(JSON.parse(readFileSync(process.argv[2], "utf-8")));
if (!r.success) {
  console.error(JSON.stringify(r.error.issues, null, 2));
  process.exit(1);
}
const bad = r.data.windows.filter((w) => w.cleaning_label !== LEVEL_STYLES[w.cleaning_level].label);
if (bad.length) {
  console.error(`cleaning_label != palette label: ${bad.map((w) => w.id).join(", ")}`);
  process.exit(1);
}
console.log(JSON.stringify(LEVEL_STYLES));
