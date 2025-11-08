# SSH Setup for Kubernetes Deployment

## Problem
The application needs to execute Helm commands on a remote Linux server (10.1.65.194) from your Windows machine.

## Solution Options

### Option 1: Use Windows OpenSSH (Recommended)

1. **Check if OpenSSH is installed:**
   ```cmd
   ssh -V
   ```
   If not installed, install via Windows Settings → Apps → Optional Features → Add OpenSSH Client

2. **Create/Copy SSH Key:**
   - Place your SSH private key at: `C:\Users\YOUR_USERNAME\.ssh\id_rsa`
   - Or specify the correct path in `routes.py` line 3193

3. **Test SSH Connection:**
   ```cmd
   ssh admin@10.1.65.194
   ```

### Option 2: Use PuTTY (Alternative)

If you prefer PuTTY:

1. Install PuTTY from: https://www.putty.org/
2. Convert your key to PPK format using PuTTYgen
3. Uncomment line 3195 in `routes.py` and update the path:
   ```python
   ssh_command = f'plink -i "C:/path/to/your/private.ppk" {ssh_host} "{helm_command}"'
   ```

### Option 3: Run App on Linux Server

Move the entire Flask application to the Linux server where microk8s is installed.

## Testing Your Setup

After configuring SSH, test by running:
```cmd
ssh admin@10.1.65.194 "microk8s kubectl get nodes"
```

If this works, the deployment will work!

## Current Configuration

- **Remote Server:** admin@10.1.65.194
- **Chart Path:** /home/admin/khalid/dsipts-p/dsipts-p-chart
- **Values Path:** /mnt/NFS/khalid/DSIPTS-P/uploads/Users/{username}/{experiment}/values_{experiment}.yaml

## Troubleshooting

**Error: 'ssh' is not recognized**
- Install OpenSSH client on Windows

**Error: Permission denied (publickey)**
- Check SSH key location
- Verify key has correct permissions
- Test manual SSH connection

**Error: Host key verification failed**
- Add the server to known hosts:
  ```cmd
  ssh admin@10.1.65.194
  ```
  Type "yes" when prompted

## Need Help?

Check the logs in the terminal where Flask is running for detailed error messages.
