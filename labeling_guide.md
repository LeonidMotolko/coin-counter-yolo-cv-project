# Labeling Guide

## Goal

Label every visible coin with one bounding box.

Class name:

```text
coin
```

## Rules

1. Draw one box around each visible coin.
2. If a coin is partly covered but still visible, label the visible full coin area as well as possible.
3. Do not label background objects.
4. Do not label shadows.
5. Keep the class name exactly:

```text
coin
```

## Recommended dataset size

Minimum:

```text
50 images
```

Better:

```text
100-300 images
```

## Recommended split

```text
80% train
20% validation
```
