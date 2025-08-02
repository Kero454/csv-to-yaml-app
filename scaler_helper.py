def get_scaler_config(request):
    """Get scaler configuration from form data."""
    return {
        'type': request.form.get('scaler_type', 'standard'),
        'feature_range': {
            'min': float(request.form.get('scaler_feature_range_min', 0)),
            'max': float(request.form.get('scaler_feature_range_max', 1))
        },
        'quantile_range': {
            'min': float(request.form.get('scaler_quantile_range_min', 25)),
            'max': float(request.form.get('scaler_quantile_range_max', 75))
        },
        'power_method': request.form.get('scaler_power_method', 'yeo-johnson'),
        'with_mean': request.form.get('scaler_with_mean') == 'on',
        'with_std': request.form.get('scaler_with_std') == 'on'
    }
