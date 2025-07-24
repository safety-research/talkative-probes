# Complete Deployment Setup Guide

This guide walks you through setting up automated deployment with GitHub Actions.

## What You Need to Do

### 1. Set Up the Submodule (One Time)

```bash
# Clone your GitHub Pages repo if you haven't already
git clone https://github.com/kitft/kitft.github.io.git ~/kitft.github.io
cd ~/kitft.github.io

# Add talkative-probes as a submodule
git submodule add https://github.com/kitft/talkative-probes.git talkative-autoencoder

# Optional: Create a cleaner URL with a symlink
ln -s talkative-autoencoder/talkative_autoencoder/website/frontend talkative-lens

# Commit and push
git add .
git commit -m "Add talkative-autoencoder submodule and symlink"
git push
```

### 2. Create Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name like "talkative-autoencoder-updater"
4. Select the `repo` scope (full control of private repositories)
5. Click "Generate token"
6. **COPY THE TOKEN NOW** - you won't see it again!

### 3. Add Token to talkative-probes Repository

1. Go to https://github.com/kitft/talkative-probes/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PAGES_UPDATE_TOKEN`
4. Value: Paste your token from step 2
5. Click "Add secret"

### 4. Copy GitHub Actions Workflows

```bash
# Create workflows directory in talkative-probes (if not exists)
mkdir -p /workspace/kitf/talkative-probes/.github/workflows
cp /workspace/kitf/talkative-probes/talkative_autoencoder/website/.github/workflows/update-github-pages.yml /workspace/kitf/talkative-probes/.github/workflows/

# Copy workflow to your github.io repo
cd ~/kitft.github.io
mkdir -p .github/workflows
cp /workspace/kitf/talkative-probes/talkative_autoencoder/website/github-pages-workflow.yml .github/workflows/update-talkative-autoencoder.yml

# Commit the workflow
git add .github/workflows/update-talkative-autoencoder.yml
git commit -m "Add auto-update workflow for talkative-autoencoder"
git push
```

### 5. Test the Setup

```bash
# Make a small change to the frontend
cd /workspace/kitf/talkative-probes/talkative_autoencoder/website
echo "<!-- Test update -->" >> frontend/index.html

# Commit and push - this should trigger the auto-update
git add frontend/index.html
git commit -m "Test auto-deployment"
git push
```

Watch the Actions tab in both repositories to see the automation in action!

## How It Works

1. When you push changes to `talkative_autoencoder/website/frontend/**` in talkative-probes
2. GitHub Action in talkative-probes triggers
3. It sends a dispatch event to your kitft.github.io repo
4. GitHub Action in kitft.github.io receives the event
5. It updates the submodule to the latest commit
6. Commits and pushes the change
7. GitHub Pages automatically deploys the update

## Accessing Your Site

With the symlink, your site will be available at:
- Full path: `https://kitft.github.io/talkative-autoencoder/talkative_autoencoder/website/frontend/`
- Clean URL: `https://kitft.github.io/talkative-lens/`

## Troubleshooting

- **Workflow not triggering?** Check the Actions tab for errors
- **Permission denied?** Ensure your PAT has `repo` scope
- **Submodule not updating?** Try `git submodule update --remote --force`