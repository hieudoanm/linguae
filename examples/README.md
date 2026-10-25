# Examples

Test the `linguae` CLI against a large set of sample texts.

## Usage

```bash
cd examples
chmod +x examples.sh
./examples.sh
```

The script builds the CLI (`cargo build --release -p linguae`) and runs it on every non-empty line in `examples.txt`, printing `text -> language`.

Agents and humans can edit `examples.txt` to add more cases.
