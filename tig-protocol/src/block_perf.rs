use
{
    std::
    {
        thread,
        sync::
        {
            Arc,
            Mutex
        },
        collections::
        {
            HashSet,
            HashMap
        }
    },
    rand::
    {
        Rng,
        RngCore,
        SeedableRng,
        rngs::
        {
            SmallRng
        }
    },
    criterion::
    {
        Criterion,
        black_box,
        criterion_group,
        criterion_main
    },
    tig_structs::
    {
        core::
        {
            Point,
            Frontier
        }
    },
    tig_utils::
    {
        FrontierOps
    },
    ndarray::
    {
        Array2,
        ArrayView2
    }
};

mod context;
mod add_block;

/*
struct BenchmarkContext
{
}

impl Context for BenchmarkContext
{
}*/

fn get_points()
                                                    -> Vec<Vec<Point>>
{
    let mut rng                                         = SmallRng::seed_from_u64(1337);
    let mut challenges                                  = Vec::new();

    for i in 0..16
    {
        let (mut min_x, mut max_x)                          = (0x7fffffff, 0);
        let (mut min_y, mut max_y)                          = (0x7fffffff, 0);

        challenges.push(Vec::new());
        let challenge_idx                                   = challenges.len() - 1;

        let base_points                                     = [[  70,  344],[  75,  336],[  98,  314],[  66,  355],[  131,  279],[  125,  281],[  68,  351],[  103,  304],[  61,  361],[  57,  369],[  92,  320],[  132,  277],[  81,  330],[  71,  340],[  71,  341],[  74,  339],[  89,  323],[  133,  276],[  67,  351],[  85,  328],[  77,  337],[  63,  360],[  77,  333],[  61,  365],[  93,  317],[  85,  325],[  57,  370],[  80,  332],[  132,  275],[  120,  288],[  68,  363],[  65,  356],[  94,  316],[  68,  349],[  108,  306],[  65,  357],[  63,  359],[  89,  321],[  90,  320],[  100,  307],[  131,  277],[  110,  298],[  118,  290],[  125,  284],[  70,  341],[  76,  334],[  75,  339],[  117,  291],[  122,  287],[  55,  375],[  73,  339],[  124,  282],[  107,  301],[  111,  294],[  121,  288],[  86,  324],[  101,  307],[  105,  304],[  118,  291],[  83,  328],[  94,  315],[  92,  316],[  97,  312],[  77,  334],[  102,  306],[  128,  279],[  125,  289],[  120,  289],[  126,  280],[  56,  372],[  73,  337],[  71,  349],[  110,  297],[  122,  286],[  119,  290],[  95,  317],[  73,  340],[  59,  366],[  79,  332],[  111,  295],[  95,  314],[  72,  339],[  108,  298],[  117,  290],[  60,  363],[  112,  294],[  106,  308],[  116,  297],[  67,  354],[  88,  323],[  91,  319],[  108,  300],[  78,  332],[  107,  300],[  86,  330],[  121,  287],[  104,  303],[  114,  292],[  61,  362],[  98,  311],[  118,  289],[  82,  328],[  62,  361],[  94,  318],[  78,  333],[  115,  291],[  58,  370],[  56,  371],[  93,  315],[  92,  319],[  105,  302],[  115,  290],[  113,  293],[  99,  308],[  91,  318],[  73,  338],[  83,  327],[  90,  322],[  77,  335],[  56,  375],[  57,  371],[  101,  306],[  69,  350],[  60,  362],[  79,  334],[  107,  299],[  105,  303],[  102,  305],[  104,  304],[  84,  328],[  93,  316],[  78,  334],[  106,  302],[  115,  301],[  70,  342],[  123,  283],[  119,  289],[  58,  368],[  70,  343],[  79,  333],[  69,  346],[  116,  290],[  99,  312],[  82,  330],[  131,  280],[  56,  373],[  88,  322],[  100,  308],[  69,  352],[  75,  335],[  71,  339],[  59,  367],[  78,  336],[  135,  275],[  131,  276],[  114,  291],[  108,  299],[  109,  299],[  112,  296],[  104,  305],[  134,  277],[  88,  325],[  106,  305],[  55,  376],[  100,  310],[  138,  275],[  69,  348],[  131,  278],[  92,  318],[  80,  333],[  71,  342],[  67,  352],[  102,  308],[  115,  292],[  111,  297],[  85,  326],[  59,  365],[  62,  360],[  92,  317],[  84,  326],[  130,  278],[  72,  338],[  85,  324],[  119,  288],[  57,  372],[  114,  294],[  126,  281],[  125,  282],[  95,  313],[  74,  338],[  110,  294],[  64,  356],[  94,  314],[  91,  317],[  95,  315],[  133,  279],[  55,  378],[  57,  373],[  121,  289],[  89,  322],[  94,  317],[  116,  291],[  109,  300],[  127,  282],[  120,  286],[  62,  364],[  90,  318],[  63,  358],[  61,  366],[  99,  309],[  76,  336],[  124,  284],[  74,  336],[  128,  280]   ];
        for point in base_points
        {
            challenges[challenge_idx].push(point.to_vec());
        }

        for point in base_points
        {
            if point[0] < min_x                             {min_x = point[0];}
            if point[0] > max_x                             {max_x = point[0];}

            if point[1] < min_y                             {min_y = point[1];}
            if point[1] > max_y                             {max_y = point[1];}
        }

        for i in 0..(base_points.len()*10)
        {
            while
            {
                let (x, y)                                  = (min_x - min_x / 2 + (rng.next_u32()%((max_x as u32 + 1) * 2)) as i32, min_y - min_y / 2 + (rng.next_u32()%((max_y as u32 + 1)) * 2) as i32);
                let mut invalid                             = true;

                for point in &challenges[challenge_idx]
                {
                    if (x - point[0]).abs() <= ((max_x - min_x)/(challenges[challenge_idx].len() as i32) / 2)
                    {
                        invalid = false;

                        break;
                    }

                    if (y - point[1]).abs() <= ((max_y - min_y)/(challenges[challenge_idx].len() as i32) / 2)
                    {
                        invalid = false;

                        break;
                    }
                }

                if !invalid
                {
                    challenges[challenge_idx].push([x, y].to_vec());
                }

                invalid
            } {}
        }
    }
    
    return challenges;
}

#[inline]
fn bench_update_qualifiers_st(
    challenges:                             &Vec<Vec<Point>>
)                                                   -> Vec<HashMap<Point, usize>>
{
    let mut frontiers                                   = Vec::new();

    for challenge_data in challenges
    {
        let points                                      = challenge_data
            .iter()
            .map(|d| d.iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
            .collect::<Frontier>();

        let mut frontier_indexes    = HashMap::<Point, usize>::new();
        for (frontier_index, frontier) in add_block::pareto_algorithm(points, false).into_iter().enumerate() 
        {
            for point in frontier 
            {
                frontier_indexes.insert(point, frontier_index);
            }
        }

        frontiers.push(frontier_indexes);
    }

    return frontiers;
}

#[inline]
fn bench_update_qualifiers_mt(
    challenges:                             &Vec<Vec<Point>>
)                                                   -> Vec<HashMap<Point, usize>>
{
    let frontiers                                       = Arc::new(Mutex::new(Vec::new()));

    thread::scope(|s|
    {
        for challenge_data in challenges
        {
            let frontiers_                              = frontiers.clone();
            s.spawn(||
            {
                let points                                      = challenge_data
                    .iter()
                    .map(|d| d.iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
                    .collect::<Frontier>();
        
                let mut frontier_indexes    = HashMap::<Point, usize>::new();
                for (frontier_index, frontier) in add_block::pareto_algorithm(points, false).into_iter().enumerate() 
                {
                    for point in frontier 
                    {
                        frontier_indexes.insert(point, frontier_index);
                    }
                }
        
                frontiers.lock().unwrap().push(frontier_indexes);
            });
        }
    });

    return Arc::try_unwrap(frontiers).unwrap().into_inner().unwrap();
}

fn get_o_pareto_points()
                                                    -> Array2<i32>
{
    let n_observations                  = 4096*2;
    let n_objectives                    = 2;
    let mut rng                         = SmallRng::seed_from_u64(1337);

    let costs: Array2<i32> = Array2::from_shape_fn((n_observations, n_objectives), |_| 
    {
        rng.gen_range(0..256)
    });

    return costs;
}

fn get_uo_pareto_points()
                                                    -> Vec<Point>
{
    let mut rng                                         = SmallRng::seed_from_u64(1337);
    let mut points                                      = Vec::new();

    for i in 0..4096*2
    {
        let (x, y)                                      = (rng.gen_range(0..256), rng.gen_range(0..256));            

        points.push([x, y].to_vec());
    }

    return points;
}

fn o_pareto_algorithm(
    points:                                 ArrayView2<i32>, 
    only_one:                               bool
)                                                   
{
    let ranks                                           = tig_utils::o_nondominated_rank(points, None);
}

fn uo_pareto_algorithm(
    points:                                 &Vec<Point>, 
    only_one:                               bool
)                                                   
{
    let points                                      = points
            .iter()
            .map(|d| d.iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
            .collect::<Frontier>();

    let ranks                                           = points.pareto_frontier();
}

pub fn criterion_benchmark(
    c:                                      &mut Criterion
) 
{
    let challenges                                      = get_points();

    /*c.bench_function("update_qualifiers_singlethread", |b|
    {
        b.iter(|| bench_update_qualifiers_st(&challenges));
    });

    c.bench_function("update_qualifiers_multithread", |b|
    {
        b.iter(|| bench_update_qualifiers_mt(&challenges));
    });*/

    let o_pareto_points                                 = get_o_pareto_points();
    c.bench_function("o_pareto_algorithm", |b|
    {
        b.iter(|| o_pareto_algorithm(o_pareto_points.view(), false));
    });

    let uo_pareto_points                                = get_uo_pareto_points();
    c.bench_function("uo_pareto_algorithm", |b|
    {
        b.iter(|| uo_pareto_algorithm(&uo_pareto_points, false));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);