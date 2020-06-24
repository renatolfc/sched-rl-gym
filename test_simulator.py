#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import pprint

from multiprocessing import Process

from lugarrl import workload
from lugarrl.scheduler import backfilling_scheduler


def simulate(trace, size):
    work = workload.TraceGenerator(
        trace,
        size,
        1024 * 1024 * 1024,
        ignore_memory=True,
    )

    scheduler = backfilling_scheduler.BackfillingScheduler(
        size, 0, ignore_memory=True
    )
    for job in work:
        while job.submission_time > scheduler.current_time:
            print(f'Fast-forwarding time to {job.submission_time}')
            try:
                scheduler.step(job.submission_time - scheduler.current_time)
            except Exception as e:
                print('Failed to step to time.', e)
        scheduler.submit(job)
    stats = scheduler.stats
    with open(trace + '.pkl', 'wb') as fp:
        pickle.dump((
            stats,
            [job.swf for job in sorted(scheduler.queue_completed, key=lambda x:x.id)]
        ), fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done processing', trace)


def main():
    traces = [('CTC-SP2-1996-3.1-cln.swf', 338), ('SDSC-SP2-1998-4.2-cln.swf', 128)]

#    for args in traces:
#        simulate(*args)

    processes = [Process(target=simulate, args=trace) for trace in traces]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()
