# GitHub 发布前检查（中文）

当前目录仅是可安装原型，不代表数据、模型或第三方依赖已获公开再分发许
可。项目已于 2026-11-06 为授权的 BRepRecover 项目代码选定 MIT，并加入
LICENSE；这不解决候选内提取行为的来源/权利审查，也不许可数据、权重、
依赖或上游代码。尚未纳入候选的基线派生文件（如 `generate.py`、
`brepBuild.py`）继续排除，除非另行完成权利审查。当前候选仍非公开发布就
绪。不要上传数据、模型、实验结果或缓存。

## 身份与风险

用户报告已完成 GitHub 登录；认证状态不在此文档中持久化，且不记录令牌或
凭据。`gh auth status` 只能确认本机 CLI 认证状态，不能证明组织授权或文件
发布权。个人公开仓库可能把账号、提交历史和本机身份信息关联起来；公开前
请自行评估去匿名化风险。代码仓库不包含 ESWA Option C 数据，不能据此声称
数据公开或数据许可已解决。

## 权利确认后的人工步骤

以下命令仅供权利与许可证确认之后的有权限维护者逐项执行；此文档不会
自动执行 Git 初始化、提交、登录或发布操作。示例假设当前目录是已经
人工审阅的候选暂存目录，不能在原研究工作区直接照抄。

1. 复核候选代码来源、贡献者授权、第三方依赖与数据/模型条款。MIT 仅适用
   于授权的 BRepRecover 项目代码，不能替代逐文件 provenance/redistribution
   review。
2. 如需验证当前本机 CLI 会话，运行 `gh auth status` 并人工核对账户与主机；
   不要将令牌、凭据或认证输出写入仓库。认证不代表权利审查通过。
3. 在经过审阅的候选暂存目录 `/home/fy/Brep/release_preparation/BRepRecover_MIT_20261106`
   检查文件列表、差异、许可证/NOTICE、密钥扫描结果和 `MANIFEST.sha256.json`，
   并复核 `docs/release_plan.md` 所列未决事项。只显式暂存已批准的文件，
   再检查 `git diff --cached`。
4. 仅将候选用于私有预览和审核；公开前必须解决剩余来源/权利、依赖和数据
   门槛并取得明确批准。Git 操作由维护者在已审阅候选副本中执行。
   不要用 `git add .`；将 `git add` 参数替换为审阅批准的逐项路径：

   ```bash
   cd /home/fy/Brep/BRepRecover
   git init -b main
   git status --short
   git add .gitignore README.md pyproject.toml requirements-core-test.txt \
     src/breprecover docs tests examples envs tools/prepare_release.py
   git diff --cached --check
   git ls-files
   git diff --cached
   # 确认暂存区只含已批准文件后再提交
   git commit -m "Prepare BRepRecover prototype"
   gh repo create BRepRecover --private --source . --remote origin
   git push -u origin HEAD
   ```

   私有远端的创建、初始化、提交和推送应由维护者单独执行。本指南不执行
   这些操作，也不意味着远端已经创建。哈希清单不应被当作权利审核证明。
   不要使用强制推送，也不要因 CLI 登录成功就改为公开仓库。
5. 远端内容经过独立复核并获得明确批准后，再由维护者决定是否改为公开。

私有仓库尚不应被描述为公众可访问；实际远端状态由维护者确认。没有任何
步骤授权发布 baseline 派生的 `generate.py`、`brepBuild.py` 或第三方数据/权重。

官方操作参考：[将本地代码加入 GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[仓库许可证说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。
