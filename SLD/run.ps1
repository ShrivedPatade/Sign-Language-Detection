# run_python.ps1
param (
    [string]$choice
)

switch ($choice) {
    "1" {
        Write-Host "Running collect_data.py..."
        python "collect_data.py"
    }
    "2" {
        Write-Host "Running create_dataset.py..."
        python "create_dataset.py"
    }
    "3" {
        Write-Host "Running train_cnn.py...."
        python "train_cnn.py"
    }
    "12" {
        Write-Host "Running both collect_data.py and create_dataset.py..."
        python "collect_data.py"
        python "create_dataset.py"
    }
    "23" {
        Write-Host "Running both create_dataset.py... and train_cnn.py..."
        python "create_dataset.py"
        python "train_cnn.py"
    }
    "123" {
        Write-Host "Running collect_data.py, create_dataset.py and train_cnn.py..."
        python "collect_data.py"
        python "create_dataset.py"
        python "train_cnn.py"
    }
    "4" {
        Write-Host "Running run_live_detection.py..."
        python "run_live_detection.py"
    }
    "34" {
        Write-Host "Running train_cnn.py and run_live_detection.py..."
        python "train_cnn.py"
        python "run_live_detection.py"
    }
    "234" {
        Write-Host "Running create_dataset.py, train_cnn.py and run_live_detection.py..."
        python "create_dataset.py"
        python "train_cnn.py"
        python "run_live_detection.py"
    }
    "1234" {
        Write-Host "Running collect_data.py, create_dataset.py, train_cnn.py and run_live_detection.py..."
        python "collect_data.py"
        python "create_dataset.py"
        python "train_cnn.py"
        python "run_live_detection.py"
    }
    Default {
        Write-Host "Invalid choice. Use: 1, 2, 12, 3, 23, 123, 4, 34, 234 or 1234."
    }
}
