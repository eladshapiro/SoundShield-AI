Run the comprehensive 5-scenario end-to-end test suite.

Run: `python3 run_5_tests.py`

This exercises all detector modules against 5 realistic synthetic audio scenarios:
1. Calm environment (no incidents expected)
2. Aggressive shouting (violence/anger expected)
3. Cry without response (neglect expected)
4. Mixed realistic (multi-phase: calm -> cry -> response -> shouting)
5. Edge cases (silence, noise, bad inputs)

Also run the integration test suite: `python3 -m pytest tests/test_integration.py -v`
This tests:
- Full detection pipeline with synthetic 10s kindergarten recording
- Config propagation to detectors
- Notification triggers from incidents
- Cry episode aggregation and response time measurement

Report which scenarios passed/failed and analyze any unexpected results by reading the relevant detector code.
