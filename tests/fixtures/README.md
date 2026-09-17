# `fixtures/` — tiny synthetic test data

Small, **synthetic** images used only by the test suite. They are *not* real
otoscopy images and carry no patient data, so they are safe to commit.

Layout mirrors a real dataset so loaders can be tested unchanged:

```
fixtures/
├── Normal/
└── Abnormal/
```

If a test needs a specific edge case (e.g. an image with no annotation, to
exercise the masking pairing logic), add a minimal synthetic example here and
document why in the test that uses it.
