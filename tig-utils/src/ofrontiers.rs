use
{
    std::
    {
        sync::
        {
            Arc,
            Mutex
        }  
    },
    crate::
    {
        Point,
        Frontier
    }
};

fn change_directions(
    frontier:                               &mut Frontier<Point>
)
{
    let objectives                                      = frontier.iter().nth(0).unwrap().len();
}

fn unique_and_inverted_indices(
    frontier:                               &mut Frontier<Point>
)
                                                    -> Vec<usize>
{
    let inverse_indices                                 = Vec::new();

    return inverse_indices;
}

fn nondominated_rank(
    frontier:                               &mut Frontier<Point>
)
{

}

fn o_pareto_frontier(
    frontier:                               &Frontier<Point>)
                                                    -> Frontier<Point>
{
    let mut frontier_                                   = Arc::new(Mutex::new(frontier.clone()));

    let total_frontiers                                 = frontier.len();
    change_directions(&mut frontier_.lock().unwrap());

    let reconstr_indices                                = unique_and_inverted_indices(&mut frontier_.lock().unwrap());  
    let ranks                                           = nondominated_rank(&mut frontier_.lock().unwrap());

    return Arc::try_unwrap(frontier_).unwrap().into_inner().unwrap();
}