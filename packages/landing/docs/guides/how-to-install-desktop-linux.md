# How to install the Linguae desktop app on Linux

For x86_64 Linux distributions.

Choose the package format that matches your distribution — all are listed under
**Linux** on the
[downloads page](https://hieudoanm.github.io/linguae/downloads/).

## Debian / Ubuntu (`.deb`)

```sh
sudo apt install ./desktop-linguae-x86_64-unknown-linux-gnu.deb
```

(or
`sudo dpkg -i desktop-linguae-x86_64-unknown-linux-gnu.deb && sudo apt --fix-broken install`)

## Fedora / RHEL / openSUSE (`.rpm`)

```sh
sudo dnf install ./desktop-linguae-x86_64-unknown-linux-gnu.rpm        # Fedora / RHEL
sudo zypper install ./desktop-linguae-x86_64-unknown-linux-gnu.rpm     # openSUSE
```

## Any distribution (`.AppImage`)

```sh
chmod +x desktop-linguae-x86_64-unknown-linux-gnu.AppImage
sudo mv desktop-linguae-x86_64-unknown-linux-gnu.AppImage /opt/linguae
sudo ln -s /opt/linguae/desktop-linguae-x86_64-unknown-linux-gnu.AppImage ~/.local/bin/linguae
```

Then run `linguae` — no installation needed, the AppImage is self-contained.

## Verifying

Launch Linguae from your application menu, or run the binary/AppImage from a
terminal.
