# Manufacturing example knowledge (synthetic demo only)

This file is synthetic demo knowledge, not a real factory specification.

For the bundled manufacturing demo, `yield_percentage` is treated as an end-of-process outcome. `processing_time_sec` and `power_consumption` may be unavailable at an earlier prediction point, so their prediction-time availability must be confirmed before deployment.

Repeated measurements from the same lot, equipment, product, patient, subject, or device should not be split randomly across train and test when the deployment target is a new entity. Use group-aware validation when such an identifier exists.

A prediction feature's importance is predictive evidence, not proof that changing that feature will causally change yield.
