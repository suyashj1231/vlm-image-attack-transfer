Traceback (most recent call last):
  File "<frozen runpy>", line 201, in _run_module_as_main
  File "<frozen runpy>", line 87, in _run_code
  File "/Users/suyashjain/Documents/Claude-code/Class/CS205/Attack/Tool/eval/summarize_transfer.py", line 146, in <module>
    main()
    ~~~~^^
  File "/Users/suyashjain/Documents/Claude-code/Class/CS205/Attack/Tool/eval/summarize_transfer.py", line 84, in main
    groups[name] = [load(p) for p in paths.split(",")]
                    ~~~~^^^
  File "/Users/suyashjain/Documents/Claude-code/Class/CS205/Attack/Tool/eval/summarize_transfer.py", line 36, in load
    with open(path, newline="") as f:
         ~~~~^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'results/grid3_attack_eot.csv'
