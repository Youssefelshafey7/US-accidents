# Git Workflow Instructions

Please follow these steps whenever you work on the project.

## 1. Before Editing Anything

Always pull the latest version of the project before making any changes:

```bash
git pull origin main
```

This makes sure you are working with the latest changes from the `main` branch.

---

## 2. Make Your Changes

After pulling the latest version, you can edit the project and make your changes.

---

## 3. Add Your Changes

When you have finished making your changes:

```bash
git add .
```

This stages all your modified and new files.

---

## 4. Commit Your Changes

Create a commit describing what you have done:

```bash
git commit -m "type a detail related to what you have done"
```

### Examples

```bash
git commit -m "add accident data preprocessing"
```

```bash
git commit -m "fix missing values handling"
```

```bash
git commit -m "update data visualization notebook"
```

```bash
git commit -m "add random forest model"
```

Try to make the commit message **short and descriptive**.

---

## 5. Push Your Changes

Finally, push your changes to GitHub:

```bash
git push
```

---

## Complete Workflow

In short, every time you work on the project:

```bash
# 1. Get the latest changes
git pull origin main

# 2. Make your changes
# ...

# 3. Stage your changes
git add .

# 4. Commit your changes
git commit -m "describe what you have done"

# 5. Push to GitHub
git push
```

## Important

**Always run `git pull origin main` before starting your work.**

This helps prevent conflicts when multiple people are working on the repository.
