# run_python.ps1
param (
    [string]$choice
)

switch ($choice) {
    "1" {
        Write-Host "Running 1_collect_data.py..."
        python "1_collect_data.py"
    }
    "2" {
        Write-Host "Running 2_create_dataset.py..."
        python "2_create_dataset.py"
    }
    "3" {
        Write-Host "Running 3_train_cnn.py...."
        python "3_train_cnn.py"
    }
    "12" {
        Write-Host "Running both 1_collect_data.py and 2_create_dataset.py..."
        python "1_collect_data.py"
        python "2_create_dataset.py"
    }
    "23" {
        Write-Host "Running both 2_create_dataset.py... and 3_train_cnn.py..."
        python "2_create_dataset.py"
        python "3_train_cnn.py"
    }
    "123" {
        Write-Host "Running 1_collect_data.py, 2_create_dataset.py and 3_train_cnn.py..."
        python "1_collect_data.py"
        python "2_create_dataset.py"
        python "3_train_cnn.py"
    }
    "4" {
        Write-Host "Running 4_run_live_detection.py..."
        python "4_run_live_detection.py"
    }
    "34" {
        Write-Host "Running 3_train_cnn.py and 4_run_live_detection.py..."
        python "3_train_cnn.py"
        python "4_run_live_detection.py"
    }
    "234" {
        Write-Host "Running 2_create_dataset.py, 3_train_cnn.py and 4_run_live_detection.py..."
        python "2_create_dataset.py"
        python "3_train_cnn.py"
        python "4_run_live_detection.py"
    }
    "1234" {
        Write-Host "Running 1_collect_data.py, 2_create_dataset.py, 3_train_cnn.py and 4_run_live_detection.py..."
        python "1_collect_data.py"
        python "2_create_dataset.py"
        python "3_train_cnn.py"
        python "4_run_live_detection.py"
    }
    Default {
        Write-Host "Invalid choice. Use: 1, 2, 12, 3, 23, 123, 4, 34, 234 or 1234."
    }
}
