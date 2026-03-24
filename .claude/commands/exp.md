Commit and push the current experiment changes to GitHub.

The user will call this as: /exp <phase> <description> [CLAP=<val>] [FAD=<val>] [other metrics]

Examples:
- /exp phase6_v2 "add noise schedule annealing" CLAP=0.1923 FAD=1.6821
- /exp phase7_v1 "curriculum learning for quality" CLAP=0.2011 FAD=1.5430 FD=12.3
- /exp phase6_v3 "fix q_embed dropout"

Steps to follow:

1. Run `git status` and `git diff --stat` to see what has changed.

2. Stage only the relevant source code files — never stage:
   - exps/, eval_output/, *.pth, *.npz, *.flac
   - *.bak, *.backup, fix_*.py, phase*/
   - Any binary or large files

   Typically stage from:
   - meanaudio/model/
   - meanaudio/data/
   - meanaudio/runner_*.py
   - meanaudio/eval_utils.py
   - eval.py, train.py, infer.py, demo.py
   - train_pipeline.sh, set_training_stage.py
   - migrate_*.py
   - .gitignore, config/

3. Build the commit message in this format:
   ```
   <phase>: <description>

   <bullet points summarizing what changed based on git diff>
   [metrics line if provided, e.g.: CLAP=0.1923 FAD=1.6821]
   ```

4. Run: `git add <files>` then `git commit -m "<message>"` then `git push`

5. Report the commit hash and confirm push succeeded.
