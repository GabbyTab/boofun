# Releasing BooFun

The release routine, written down so it is mechanical every time (issue #60).
Reference precedent: [v1.3.0](https://github.com/GabbyTab/boofun/releases/tag/v1.3.0)
and its [announcement post](https://github.com/GabbyTab/boofun/discussions/63).

## Checklist

1. **Pick the version** — [SemVer](https://semver.org): patch for fixes, minor
   for features, major for breaking changes.
2. **Update the version** in `pyproject.toml` (`[project] version`, keep the
   release-date comment current).
3. **Update `CHANGELOG.md`** — [Keep a Changelog](https://keepachangelog.com)
   format: a `## [X.Y.Z] - YYYY-MM-DD` section with a one-paragraph theme
   summary, then `Added` / `Changed` / `Fixed` / `Removed` subsections.
4. **Open a release PR** with both changes; merge on green CI.
5. **Tag from main**:

   ```bash
   git switch main && git pull
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

   The tag triggers the full OS × Python test matrix plus docs, and — only if
   those pass — the `publish` job: build, Sigstore signing, and PyPI upload
   (see the `publish` job in `.github/workflows/ci.yml`). No manual upload.
6. **Verify PyPI**: `pip index versions boofun` shows the new version
   (allow a few minutes).
7. **Create the GitHub Release** for the tag. Title style:
   `vX.Y.Z — the <theme> release`. Body: the CHANGELOG section for this
   version, plus a link to the announcement post (added in step 9).
8. **Post the announcement** in
   [Discussions → Announcements](https://github.com/GabbyTab/boofun/discussions/categories/announcements)
   using the template below.
9. **Cross-link**: edit the GitHub Release body to link the discussion, and
   make sure the discussion links the release and CHANGELOG.

## Timing

Announcements may notify watchers, so post them on **weekends** (maintainer
preference). The tag/PyPI publish itself can happen any time; only the
announcement is timing-sensitive.

## Announcement template

Matches the v1.3.0 post so precedent and template agree:

````markdown
BooFun **vX.Y.Z** is out! 🎉

```bash
pip install --upgrade boofun
```

<One-paragraph theme: what this release is about and why it matters.>

- **<Highlight 1>** — <one or two sentences>
- **<Highlight 2>** — ...
- **<Highlight 3>** — ...

**Breaking changes**: <none, or a list with upgrade notes>

**Upgrade notes**: <anything a user must do; omit if nothing>

Full details: [CHANGELOG](https://github.com/GabbyTab/boofun/blob/main/CHANGELOG.md)
· [Release notes](https://github.com/GabbyTab/boofun/releases/tag/vX.Y.Z)

Feedback and bug reports welcome — [issues](https://github.com/GabbyTab/boofun/issues)
· [discussions](https://github.com/GabbyTab/boofun/discussions).
````

## Notes

- The `publish` job requires the `PYPI_TOKEN` repository secret; if it is
  missing the job skips the upload with a warning rather than failing.
- If the tag build goes red, fix on `main`, delete the tag
  (`git push origin :refs/tags/vX.Y.Z`), and re-tag — never publish from a
  red build.
- Conda packaging (`conda-recipe/`) is not part of this routine yet.
