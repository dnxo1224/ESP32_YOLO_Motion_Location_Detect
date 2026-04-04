const fs = require('fs');
const nb = JSON.parse(fs.readFileSync('WiDR_full_pipeline.ipynb', 'utf8'));

let codeNum = 0;
for (const cell of nb.cells) {
    if (cell.cell_type === 'code') {
        codeNum++;
        const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
        console.log('\n' + '='.repeat(80));
        console.log(`CODE CELL #${codeNum}`);
        console.log('='.repeat(80));
        console.log(source);
    }
}
