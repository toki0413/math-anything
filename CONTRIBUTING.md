# 贡献指南 | Contributing Guide

感谢您对 Math Anything 项目的关注！我们欢迎来自 GitHub 和 Gitee 社区的贡献。

Thanks for your interest in Math Anything! We welcome contributions from both GitHub and Gitee communities.

---

## 🌐 双平台仓库 | Dual Platform Repositories

本项目同时在两个平台维护，您可以选择访问速度更快的平台：

This project is maintained on both platforms. Choose the one with better access speed:

| 平台 Platform | 地址 URL | 适用场景 Best for |
|--------------|----------|------------------|
| **Gitee** | https://gitee.com/crested-ibis-0413/math-anything | 国内用户 China users |
| **GitHub** | https://github.com/toki0413/math-anything | 国际用户 International users |

两个仓库保持同步，您可以在任一平台提交 Issue 或 PR。
Both repositories are synchronized. You can submit issues or PRs on either platform.

---

## 🚀 快速开始 | Quick Start

### 1. Fork 仓库 | Fork the Repository

**GitHub 用户 | GitHub Users:**
```bash
git clone https://github.com/toki0413/math-anything.git
cd math-anything
```

**Gitee 用户 | Gitee Users:**
```bash
git clone https://gitee.com/crested-ibis-0413/math-anything.git
cd math-anything
```

### 2. 安装依赖 | Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .  # 开发模式安装 | Install in development mode
```

### 3. 创建分支 | Create Branch

```bash
git checkout -b feature/your-feature-name
# 或 bugfix/issue-description
```

---

## 📝 提交 Issue | Submitting Issues

### 报告 Bug | Bug Reports

请使用 Bug Report 模板，并包含以下信息：

Please use the Bug Report template and include:

- **问题描述 | Description**: 清晰简洁的描述 Clear, concise description
- **复现步骤 | Steps to Reproduce**: 详细的操作步骤 Detailed steps
- **期望行为 | Expected Behavior**: 应该发生什么 What should happen
- **实际行为 | Actual Behavior**: 实际发生了什么 What actually happened
- **环境信息 | Environment**:
  - Python 版本 | Python version
  - 操作系统 | Operating system
  - 相关引擎 | Affected engine (VASP/LAMMPS/etc.)
- **错误日志 | Error Logs**: 完整的错误信息 Complete error messages

### 功能建议 | Feature Requests

请使用 Feature Request 模板，并说明：

Please use the Feature Request template and explain:

- **功能描述 | Feature Description**: 想要的功能 What you want
- **使用场景 | Use Case**: 为什么需要这个功能 Why you need it
- **实现建议 | Proposed Solution**: 可能的实现方式 How it might work (可选 optional)

---

## 🔧 开发流程 | Development Workflow

### 代码规范 | Code Style

- 遵循 PEP 8 规范 | Follow PEP 8
- 使用 4 空格缩进 | Use 4-space indentation
- 最大行长度 100 字符 | Max line length 100 characters
- 使用类型注解 | Use type hints where appropriate

```bash
# 格式化代码 | Format code
black math-anything/

# 检查代码风格 | Check style
flake8 math-anything/
```

### 提交规范 | Commit Messages

使用清晰的提交信息：

Use clear commit messages:

```
feat: 添加新引擎支持 | Add support for new engine
test: 添加测试用例 | Add test cases
docs: 更新文档 | Update documentation
fix: 修复参数解析错误 | Fix parameter parsing bug
refactor: 重构提取器 | Refactor extractor
```

### 测试 | Testing

提交前请确保所有测试通过：

Please ensure all tests pass before submitting:

```bash
# 运行所有测试 | Run all tests
pytest

# 运行特定测试 | Run specific tests
pytest math-anything/core/tests/test_schema.py

# 带覆盖率报告 | With coverage report
pytest --cov=math_anything
```

---

## 📤 提交 PR | Submitting Pull Requests

### PR 流程 | PR Process

1. **确保测试通过 | Ensure tests pass**
   ```bash
   pytest
   ```

2. **更新文档 | Update documentation**
   - 如果修改了 API，请更新相关文档
   - If you modified APIs, update relevant documentation

3. **填写 PR 模板 | Fill in the PR template**
   - 描述变更内容 Describe the changes
   - 关联相关 Issue Link related issues
   - 说明测试情况 Explain testing status

4. **等待审查 | Wait for review**
   - 维护者会在 3-5 个工作日内回复
   - Maintainers will respond within 3-5 business days

### PR 标题规范 | PR Title Format

```
[引擎] 简短描述 | [Engine] Brief description

示例 | Examples:
- [VASP] 添加 ENCUT 参数验证 | Add ENCUT parameter validation
- [CORE] 修复数学模式推断错误 | Fix math schema inference bug
- [DOCS] 更新 README 安装说明 | Update README installation guide
```

---

## 🏗️ 项目结构 | Project Structure

```
math-anything/
├── math-anything/
│   ├── core/              # 核心功能 | Core functionality
│   ├── *-harness/         # 引擎适配器 | Engine harnesses
│   └── skill/             # Skill 定义 | Skill definitions
├── examples/              # 使用示例 | Usage examples
├── tests/                 # 测试文件 | Test files
├── docs/                  # 文档 | Documentation
└── README.md
```

---

## 💬 联系我们 | Contact

- **GitHub Issues**: https://github.com/toki0413/math-anything/issues
- **Gitee Issues**: https://gitee.com/crested-ibis-0413/math-anything/issues
- **讨论区 Discussions**: 欢迎在 Issue 中发起讨论 | Feel free to start discussions in Issues

---

## 📜 行为准则 | Code of Conduct

- 保持友好和尊重 | Be friendly and respectful
- 欢迎新手提问 | Welcome questions from beginners
- 建设性反馈 | Provide constructive feedback
- 关注技术本身 | Focus on technical merit

---

再次感谢您的贡献！

Thank you again for your contribution!
