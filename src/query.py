def get_cummulative_reward():
    return f"""
    SELECT
        g.simulation_group,
        s.episode,
        CAST(LTRIM(s.building_name, 'Building_') AS INTEGER) AS building_id,
        AVG(s.value) AS value
    FROM (
        SELECT
            s.simulation_id,
            s.episode,
            s.building_name,
            SUM(s.reward) AS value
        FROM detailed_summary s
        GROUP BY
            s.simulation_id,
            s.episode,
            s.building_name
    ) s
    LEFT JOIN grid g ON
        g.simulation_id = s.simulation_id
    GROUP BY
        g.simulation_group,
        s.episode,
        s.building_name
    """

def get_building_cost_summary():
    return """
    SELECT
        g.simulation_group,
        d.episode,
        CAST(LTRIM(d.building_name, 'Building_') AS INTEGER) AS building_id,
        d.cost,
        AVG(d.value) AS value
    FROM (
        SELECT
            d.simulation_id,
            d.episode,
            d.building_name,
            'electricity_consumption' AS cost,
            SUM(MAX(0, d.net_electricity_consumption))/SUM(MAX(0, d.net_electricity_consumption_without_storage)) AS value
        FROM detailed_summary d
        GROUP BY
            d.simulation_id,
            d.episode,
            d.building_name

        UNION ALL

        SELECT
            d.simulation_id,
            d.episode,
            d.building_name,
            'carbon_emission' AS cost,
            SUM(MAX(0, d.net_electricity_consumption_emission))/SUM(MAX(0, d.net_electricity_consumption_emission_without_storage)) AS value
        FROM detailed_summary d
        GROUP BY
            d.simulation_id,
            d.episode,
            d.building_name

        UNION ALL

        SELECT
            d.simulation_id,
            d.episode,
            d.building_name,
            'price' AS cost,
            SUM(MAX(0, d.net_electricity_consumption_price))/SUM(MAX(0, d.net_electricity_consumption_price_without_storage)) AS value
        FROM detailed_summary d
        GROUP BY
            d.simulation_id,
            d.episode,
            d.building_name

        UNION ALL

        SELECT
            t.simulation_id,
            t.episode,
            t.building_name,
            'ramping' AS cost,
            SUM(t.with_storage_value)/SUM(without_storage_value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                d.building_name,
                ABS(d.net_electricity_consumption - LAG(d.net_electricity_consumption, 1) OVER (
                    PARTITION BY d.simulation_id, d.episode, d.building_id ORDER BY d.timestamp ASC
                )) AS with_storage_value,
                ABS(d.net_electricity_consumption_without_storage - LAG(d.net_electricity_consumption_without_storage, 1) OVER (
                    PARTITION BY d.simulation_id, d.episode, d.building_id ORDER BY d.timestamp ASC
                )) AS without_storage_value
            FROM detailed_summary d
        ) t
        GROUP BY
            t.simulation_id,
            t.episode,
            t.building_name

        UNION ALL

        SELECT
            t.simulation_id,
            t.episode,
            t.building_name,
            'daily_peak' AS cost,
            AVG(with_storage_value)/AVG(without_storage_value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                d.building_name,
                MAX(d.net_electricity_consumption) AS with_storage_value,
                MAX(d.net_electricity_consumption_without_storage) AS without_storage_value
            FROM detailed_summary d
            GROUP BY
                d.simulation_id,
                d.episode,
                d.building_name,
                DATE(d.timestamp)
        ) t
        GROUP BY
            t.simulation_id,
            t.episode,
            t.building_name

        UNION ALL

        SELECT
            t.simulation_id,
            t.episode,
            t.building_name,
            'load_factor' AS cost,
            AVG(with_storage_value)/AVG(without_storage_value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                d.building_name,
                1 - (AVG(d.net_electricity_consumption)/MAX(d.net_electricity_consumption)) AS with_storage_value,
                1 - (AVG(d.net_electricity_consumption_without_storage)/MAX(d.net_electricity_consumption_without_storage)) AS without_storage_value
            FROM detailed_summary d
            GROUP BY
                d.simulation_id,
                d.episode,
                d.building_name,
                STRFTIME('%Y', d.timestamp),
                STRFTIME('%m', d.timestamp)
        ) t
        GROUP BY
            t.simulation_id,
            t.episode,
            t.building_name

        UNION ALL

        SELECT
            t.simulation_id,
            t.episode,
            t.building_name,
            'zero_net_energy' AS cost,
            AVG(t.with_storage_value)/AVG(t.without_storage_value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                d.building_name,
                SUM(d.net_electricity_consumption) AS with_storage_value,
                SUM(d.net_electricity_consumption_without_storage) AS without_storage_value
            FROM detailed_summary d
            GROUP BY
                d.simulation_id,
                d.episode,
                d.building_name,
                STRFTIME('%Y', d.timestamp)
        ) t
        GROUP BY
            t.simulation_id,
            t.episode,
            t.building_name
    ) d
    LEFT JOIN grid g ON
        g.simulation_id = d.simulation_id
    GROUP BY
        g.simulation_group,
        d.episode,
        d.building_name,
        d.cost
    """

def get_district_cost_summary():
    return f"""
    WITH s AS (
        SELECT
            d.simulation_id,
            d.episode,
            d.timestamp,
            SUM(d.net_electricity_consumption) AS net_electricity_consumption,
            SUM(d.net_electricity_consumption_without_storage) AS net_electricity_consumption_without_storage
        FROM detailed_summary d
        GROUP BY
            d.simulation_id,
            d.episode,
            d.timestamp
    )

    SELECT
        g.simulation_group,
        d.episode,
        d.cost,
        AVG(d.value) AS value
    FROM (
        SELECT
            d.simulation_id,
            d.episode,
            'electricity_consumption' AS cost,
            AVG(d.value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                SUM(MAX(0, d.net_electricity_consumption))/SUM(MAX(0, d.net_electricity_consumption_without_storage)) AS value
            FROM detailed_summary d
            GROUP BY
                d.simulation_id,
                d.episode,
                d.building_name
         ) d
        GROUP BY
            d.simulation_id,
            d.episode

        UNION ALL

        SELECT
            d.simulation_id,
            d.episode,
            'carbon_emission' AS cost,
            AVG(d.value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                SUM(MAX(0, d.net_electricity_consumption_emission))/SUM(MAX(0, d.net_electricity_consumption_emission_without_storage)) AS value
            FROM detailed_summary d
            GROUP BY
                d.simulation_id,
                d.episode,
                d.building_name
        ) d
        GROUP BY
            d.simulation_id,
            d.episode

        UNION ALL

        SELECT
            d.simulation_id,
            d.episode,
            'price' AS cost,
            AVG(d.value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                SUM(MAX(0, d.net_electricity_consumption_price))/SUM(MAX(0, d.net_electricity_consumption_price_without_storage)) AS value
            FROM detailed_summary d
            GROUP BY
                d.simulation_id,
                d.episode,
                d.building_name
        ) d
        GROUP BY
            d.simulation_id,
            d.episode

        UNION ALL

        SELECT
            t.simulation_id,
            t.episode,
            'ramping' AS cost,
            SUM(t.with_storage_value)/SUM(without_storage_value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                ABS(d.net_electricity_consumption - LAG(d.net_electricity_consumption, 1) OVER (
                    PARTITION BY d.simulation_id, d.episode ORDER BY d.timestamp ASC
                )) AS with_storage_value,
                ABS(d.net_electricity_consumption_without_storage - LAG(d.net_electricity_consumption_without_storage, 1) OVER (
                    PARTITION BY d.simulation_id, d.episode ORDER BY d.timestamp ASC
                )) AS without_storage_value
            FROM s d
        ) t
        GROUP BY
            t.simulation_id,
            t.episode

        UNION ALL

        SELECT
            t.simulation_id,
            t.episode,
            'daily_peak' AS cost,
            AVG(with_storage_value)/AVG(without_storage_value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                MAX(d.net_electricity_consumption) AS with_storage_value,
                MAX(d.net_electricity_consumption_without_storage) AS without_storage_value
            FROM s d
            GROUP BY
                d.simulation_id,
                d.episode,
                DATE(d.timestamp)
        ) t
        GROUP BY
            t.simulation_id,
            t.episode

        UNION ALL

        SELECT
            t.simulation_id,
            t.episode,
            'load_factor' AS cost,
            AVG(with_storage_value)/AVG(without_storage_value) AS value
        FROM (
            SELECT
                d.simulation_id,
                d.episode,
                1 - (AVG(d.net_electricity_consumption)/MAX(d.net_electricity_consumption)) AS with_storage_value,
                1 - (AVG(d.net_electricity_consumption_without_storage)/MAX(d.net_electricity_consumption_without_storage)) AS without_storage_value
            FROM s d
            GROUP BY
                d.simulation_id,
                d.episode,
                STRFTIME('%Y', d.timestamp),
                STRFTIME('%m', d.timestamp)
        ) t
        GROUP BY
            t.simulation_id,
            t.episode

        UNION ALL

        SELECT
            t.simulation_id,
            t.episode,
            'zero_net_energy' AS cost,
            AVG(t.value) AS value
        FROM (
            SELECT
                t.simulation_id,
                t.episode,
                AVG(t.with_storage_value)/AVG(t.without_storage_value) AS value
            FROM (
                SELECT
                    d.simulation_id,
                    d.episode,
                    d.building_name,
                    SUM(d.net_electricity_consumption) AS with_storage_value,
                    SUM(d.net_electricity_consumption_without_storage) AS without_storage_value
                FROM detailed_summary d
                GROUP BY
                    d.simulation_id,
                    d.episode,
                    d.building_name,
                    STRFTIME('%Y', d.timestamp)
            ) t
            GROUP BY
                t.simulation_id,
                t.episode,
                t.building_name
        ) t
        GROUP BY
            t.simulation_id,
            t.episode
    ) d
    LEFT JOIN grid g ON
        g.simulation_id = d.simulation_id
    GROUP BY
        g.simulation_group,
        d.episode,
        d.cost
    """

def get_building_average_daily_profile():
    return """
    SELECT
        d.hour,
        d.episode,
        CAST(LTRIM(d.building_name, 'Building_') AS INTEGER) AS building_id,
        AVG(d.with_storage_value) AS with_storage_value,
        AVG(d.without_storage_value) AS without_storage_value
    FROM (
        SELECT
            CAST (STRFTIME('%H', d.timestamp) AS INTEGER) AS hour,
            d.episode,
            d.building_name,
            AVG(d.net_electricity_consumption) AS with_storage_value,
            AVG(d.net_electricity_consumption_without_storage) AS without_storage_value
        FROM detailed_summary d
        GROUP BY
            d.simulation_id,
            d.episode,
            d.building_name,
            STRFTIME('%H', d.timestamp)
    ) d
    GROUP BY
        d.hour,
        d.episode,
        d.building_name
    """

def get_district_average_daily_profile():
    return """
    SELECT
        d.hour,
        d.episode,
        AVG(d.with_storage_value) AS with_storage_value,
        AVG(d.without_storage_value) AS without_storage_value
    FROM (
        SELECT
            hour,
            d.episode,
            AVG(d.with_storage_value) AS with_storage_value,
            AVG(d.without_storage_value) AS without_storage_value
        FROM (
            SELECT
                d.simulation_id,
                CAST (STRFTIME('%H', d.timestamp) AS INTEGER) AS hour,
                d.episode,
                SUM(d.net_electricity_consumption) AS with_storage_value,
                SUM(d.net_electricity_consumption_without_storage) AS without_storage_value
            FROM detailed_summary d
            GROUP BY
                d.simulation_id,
                d.episode,
                d.timestamp
        ) d
        GROUP BY
            d.simulation_id,
            d.episode,
            d.hour
    ) d
    GROUP BY
        d.hour,
        d.episode
    """